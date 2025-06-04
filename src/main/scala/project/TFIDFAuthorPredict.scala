package project

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.math.log10
import java.nio.file.Paths

object TFIDFAuthorPredict {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("TFIDF Author Prediction").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val knownPath = "control/federalist_papers_control_5"
    val unknownPath = "control/federalist_papers_control_unknown_5"

    val knownDocsRaw = sc.wholeTextFiles(knownPath)
    val knownDocs = knownDocsRaw.map { case (path, content) =>
      val lines = content.split("\r?\n").map(_.trim).filter(_.nonEmpty)
      val author = lines.head.toUpperCase
      val text = lines.tail.mkString(" ")
        .toLowerCase.replaceAll("[^a-z\\s]", "").split("\\s+").filter(_.nonEmpty).toSeq
      val docId = Paths.get(path).getFileName.toString
      (docId, (author, text))
    }.cache()

    val knownAuthors = knownDocs.map { case (docId, (author, _)) => (docId, author) }.collectAsMap()

    val unknownDocs = sc.wholeTextFiles(unknownPath).map { case (path, content) =>
      val text = content.split("\r?\n").mkString(" ")
        .toLowerCase.replaceAll("[^a-z\\s]", "").split("\\s+").filter(_.nonEmpty).toSeq
      val docId = Paths.get(path).getFileName.toString
      (docId, text)
    }.cache()

    val allDocs = knownDocs.map { case (id, (_, tokens)) => (id, tokens) }.union(unknownDocs)
    val numDocs = allDocs.count()

    val termFreqs = allDocs.flatMap { case (docId, tokens) =>
      val termCounts = tokens.groupBy(identity).mapValues(_.size.toDouble)
      val maxTf = termCounts.values.max
      termCounts.map { case (term, count) => ((docId, term), count / maxTf) }
    }

    val docFreqs = allDocs.flatMap { case (docId, tokens) => tokens.distinct.map(term => (term, docId)) }
      .distinct()
      .map { case (term, _) => (term, 1) }
      .reduceByKey(_ + _)

    val idfs = docFreqs.map { case (term, df) => (term, log10(numDocs.toDouble / df)) }.cache()

    val tfidf = termFreqs.map { case ((docId, term), tf) => (term, (docId, tf)) }
      .join(idfs.map { case (term, idf) => (term, idf) })
      .map { case (term, ((docId, tf), idf)) => ((docId, term), tf * idf) }

    val tfidfVectors = tfidf.map { case ((docId, term), tfidf) => (docId, (term, tfidf)) }
      .groupByKey()
      .mapValues(_.toMap)
      .cache()

    val knownVecs = tfidfVectors.filter { case (docId, _) => knownAuthors.contains(docId) }.cache()
    val unknownVecs = tfidfVectors.filter { case (docId, _) => !knownAuthors.contains(docId) }.cache()

    val similarities = unknownVecs.cartesian(knownVecs).map { case ((uId, uVec), (kId, kVec)) =>
      val dot = uVec.keySet.intersect(kVec.keySet).map(t => uVec(t) * kVec(t)).sum
      val magU = math.sqrt(uVec.values.map(x => x * x).sum)
      val magK = math.sqrt(kVec.values.map(x => x * x).sum)
      val sim = if (magU != 0 && magK != 0) dot / (magU * magK) else 0.0
      ((uId, kId), sim)
    }

    val topMatches = similarities.map { case ((uId, kId), sim) => (uId, (kId, sim)) }
      .groupByKey()
      .mapValues(_.toList.sortBy(-_._2).take(5))
      .cache()

    // New: Soft-labeling and co-authorship logic
    val softPredictions = topMatches.mapValues { neighbors =>
      val scoreMap = neighbors.map { case (kId, sim) => (knownAuthors(kId), sim) }
        .groupBy(_._1)
        .mapValues(_.map(_._2).sum)

      val total = scoreMap.values.sum
      val percentages = scoreMap.mapValues(score => score / total)

      val sorted = percentages.toList.sortBy(-_._2)
      val top = sorted.head
      val label = if (top._2 >= 0.6) top._1 else "POSSIBLE CO-AUTHORSHIP"
      (label, sorted)
    }

    println("\nSoft-labeled predictions:")
    softPredictions.collect().foreach { case (uId, (label, scores)) =>
      println(f"\n$uId -> $label")
      scores.foreach { case (author, pct) =>
        println(f"  $author%-10s: ${(pct * 100).formatted("%.2f")}%%")
      }
    }

    println("\nTop matches per unknown doc:")
    topMatches.collect().foreach { case (uId, neighbors) =>
      println(s"\nTop matches for $uId:")
      neighbors.foreach { case (kId, sim) =>
        println(f"  $kId%-30s similarity: $sim%.4f (${knownAuthors(kId)})")
      }
    }

    sc.stop()
  }
}
