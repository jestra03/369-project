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

    val knownPath = "federalist_papers"
    val unknownPath = "federalist_papers_unknown"

    // Load known documents as (docId, (author, tokens))
    val knownDocsRaw = sc.wholeTextFiles(knownPath)
    val knownDocs = knownDocsRaw.map { case (path, content) =>
      val lines = content.split("\r?\n").map(_.trim).filter(_.nonEmpty)
      val author = lines.head.toUpperCase
      val text = lines.tail.mkString(" ")
        .toLowerCase.replaceAll("[^a-z\\s]", "").split("\\s+").filter(_.nonEmpty).toSeq
      val docId = Paths.get(path).getFileName.toString
      (docId, (author, text))
    }.cache()

    // Extract (docId, author) for later comparison
    val knownAuthors = knownDocs.map { case (docId, (author, _)) => (docId, author) }.collectAsMap()

    // Load unknown documents as (docId, tokens)
    val unknownDocs = sc.wholeTextFiles(unknownPath).map { case (path, content) =>
      val text = content.split("\r?\n").mkString(" ")
        .toLowerCase.replaceAll("[^a-z\\s]", "").split("\\s+").filter(_.nonEmpty).toSeq
      val docId = Paths.get(path).getFileName.toString
      (docId, text)
    }.cache()

    // Combine all docs into a single RDD of (docId, tokens)
    val allDocs = knownDocs.map { case (id, (_, tokens)) => (id, tokens) }.union(unknownDocs)
    val numDocs = allDocs.count()

    // Compute term frequency: ((docId, term), tf)
    val termFreqs = allDocs.flatMap { case (docId, tokens) =>
      val termCounts = tokens.groupBy(identity).mapValues(_.size.toDouble)
      val maxTf = termCounts.values.max
      termCounts.map { case (term, count) => ((docId, term), count / maxTf) }
    }

    // Document frequency: (term, docFreq)
    val docFreqs = allDocs.flatMap { case (docId, tokens) => tokens.distinct.map(term => (term, docId)) }
      .distinct()
      .map { case (term, _) => (term, 1) }
      .reduceByKey(_ + _)

    // Compute IDF values: (term, idf)
    val idfs = docFreqs.map { case (term, df) => (term, log10(numDocs.toDouble / df)) }.cache()

    // Join TF with IDF: ((docId, term), tfidf)
    val tfidf = termFreqs.map { case ((docId, term), tf) => (term, (docId, tf)) }
      .join(idfs.map { case (term, idf) => (term, idf) })  // Ensure both sides have (term, value) structure
      .map { case (term, ((docId, tf), idf)) => ((docId, term), tf * idf) }

    // Convert TF-IDF into vector: (docId, Map(term -> tfidf))
    val tfidfVectors = tfidf.map { case ((docId, term), tfidf) => (docId, (term, tfidf)) }
      .groupByKey()
      .mapValues(_.toMap)
      .cache()

    // Split known and unknown again
    val knownVecs = tfidfVectors.filter { case (docId, _) => knownAuthors.contains(docId) }.cache()
    val unknownVecs = tfidfVectors.filter { case (docId, _) => !knownAuthors.contains(docId) }.cache()

    // Compute cosine similarity between each unknown doc and each known doc
    val similarities = unknownVecs.cartesian(knownVecs).map { case ((uId, uVec), (kId, kVec)) =>
      val dot = uVec.keySet.intersect(kVec.keySet).map(t => uVec(t) * kVec(t)).sum
      val magU = math.sqrt(uVec.values.map(x => x * x).sum)
      val magK = math.sqrt(kVec.values.map(x => x * x).sum)
      val sim = if (magU != 0 && magK != 0) dot / (magU * magK) else 0.0
      ((uId, kId), sim)
    }

    // For each unknown document, get top 5 most similar known docs
    val topMatches = similarities.map { case ((uId, kId), sim) => (uId, (kId, sim)) }
      .groupByKey()
      .mapValues(_.toList.sortBy(-_._2).take(5))
      .cache()

    // Predict author by majority vote
    val predictions = topMatches.mapValues { neighbors =>
      val authorVotes = neighbors.map { case (kId, _) => knownAuthors(kId) }
      authorVotes.groupBy(identity).mapValues(_.size).toList.sortBy(-_._2).head._1
    }

    // Display predictions
    println("\nPredicted authors:")
    predictions.collect().foreach { case (uId, author) =>
      println(f"$uId -> $author")
    }

    // Display top matches as well
    topMatches.collect().foreach { case (uId, neighbors) =>
      println(s"\nTop matches for $uId:")
      neighbors.foreach { case (kId, sim) =>
        println(f"  $kId%-30s similarity: $sim%.4f (${knownAuthors(kId)})")
      }
    }

    sc.stop()
  }
}