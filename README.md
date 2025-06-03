# TF-IDF Author Attribution (Federalist Papers)

This Spark Scala project performs **authorship prediction** on the Federalist Papers using a TF-IDF + cosine similarity approach. The code processes both known and disputed papers, then predicts the likely author based on stylistic similarity.

## Project Structure

This is stored within a `lab3` or `lab` directory in some setups, but the **actual project root is the `369-project` folder**.

```
369-project/
├── build.sbt
├── project/
│   └── build.properties
├── src/
│   └── main/
│       └── scala/
│           └── project/
│               └── TFIDFAuthorPredict.scala
├── federalist_papers/
│   └── federalist_01.txt ...
├── federalist_papers_unknown/
│   └── federalist_02.txt ...
```

## How to Run

1. **Ensure dependencies are set up:**

   - Scala 2.11.8
   - Spark 2.4.8
   - Java 8

2. **Create input folders:**

Place your input text files into these folders:

- `federalist_papers/` for known papers. Each file should have the author in the first line (e.g., HAMILTON, MADISON, JAY).
- `federalist_papers_unknown/` for the papers whose authorship will be predicted.

3. **Compile and run:**

Navigate to the root project folder:

```bash
cd path/to/369-project
sbt compile
sbt run
```

4. **Output:**

Predicted authors for the unknown papers will be printed to the console, along with the top 5 most similar known papers for each.

## Notes

- This project uses RDDs and avoids external libraries to compute TF-IDF from scratch.
- Make sure the input directory paths are correct relative to the root (`369-project`).
- Works with local[*] Spark master mode.

## Example

```
Predicted authors:
federalist_10.txt -> MADISON
federalist_02.txt -> JAY
federalist_12.txt -> HAMILTON
```