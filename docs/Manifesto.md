# Manifesto

## Origins

The phrase "mechanical sympathy" was coined by Jackie Stewart, a legendary
Formula 1 world champion. He used the term to describe the need for a racing
driver to have an intuitive, deep understanding of how a car works in order to
drive it effectively, treating the machine with respect to get the best
performance.

The term was later applied to software development by Martin Thompson, who used
it to argue that developers should understand the underlying hardware to write
efficient code.

## What It Means Here

It means building software that takes into account how computers actually work:
hardware level, OS level, and system libraries. Instead of throwing piles of
hardware at a problem, one needs to design software that runs with the least
friction, the least unnecessary work, and maximum flow.

I am a big fan of this philosophy. When people try to squeeze vector data into
pages of a B+-tree storage engine or layers of LSM-tree, I can understand the
business necessity of such technical decisions, but "it insults my
intelligence."

## Why Sketch2 Exists

The nice thing about doing greenfield development without business constraints
is that you can experiment with the best possible design decisions. That is
what I am doing here.

I am building a vector storage engine with a built-in compute layer. It is a
library that can be integrated with existing databases and provide the best
technical solution for storing vectors and running specialized computations on
this data.

The vector data is stored in a way that minimizes load on the storage layer so
it can achieve high I/O performance. The vectors are laid out in memory to
allow frictionless flow of data to CPU caches for processing. They are aligned
in memory in a way that allows efficient SIMD operations on the vector data.

There are "compute units" that use the most efficient SIMD operations on
different platforms. It runs on Intel CPUs, AMD CPUs, and ARM-based CPUs. It
can select the appropriate set of instructions to get the best out of the
available hardware.

## Integrations And Direction

There is an initial integration with Python and SQLite. It is possible to
"play" with the software in Python scripts and integrate vector data processing
with queries on other data in SQLite tables.

The plan is to build integrations with Postgres and MySQL as well, maybe
integration with FAISS too, though that is not certain yet.

At this stage, the processing is built around brute-force crunching of vector
data without indexes. The next stage, after building the base
high-performance mechanisms, will be implementing IVF and PQ indexing.

## Bottom Line

Bottom line: "storing vector data in B+-tree blobs" is inhumane. I am trying
to figure out how to store and process vector data with mechanical sympathy.
