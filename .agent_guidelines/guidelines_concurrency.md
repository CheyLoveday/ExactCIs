Guideline: Concurrency & Asynchronicity Audit
Primary Directive: Using the insights from the profiling analysis, perform a read-only scan of the codebase to identify and report specific opportunities for concurrent or asynchronous execution.

Critical Constraint: DO NOT MODIFY any project source code. All findings must be documented in a report file as specified in guidelines-reporting.md.

Audit Steps
1. Target I/O-Bound Operations for Asynchronicity
Code Pattern: Look for for loops that contain I/O-bound function calls identified during profiling (e.g., making sequential API calls, querying a database, or reading files one by one).

Data Independence: Verify that the operations within the loop are independent (i.e., the result of one iteration is not required for the next).

Recommendation:

Report: List the exact file and function containing the sequential I/O loop.

Suggest: Propose refactoring the code to use Python's asyncio library. Recommend converting the function to async def and using asyncio.gather() to run the I/O operations concurrently.

2. Target CPU-Bound Operations for Parallelism
Code Pattern: Look for functions or loops that perform heavy, repetitive calculations on independent chunks of data. These should correspond to the CPU-bound bottlenecks from the profiling report.

Data Independence: Confirm that the processing of each data chunk does not depend on the results of other chunks.

Recommendation:

Report: List the exact file and function containing the CPU-intensive loop or operation.

Suggest: Propose using Python's multiprocessing module, specifically multiprocessing.Pool, to distribute the computational work across multiple CPU cores.

3. Identify General Concurrency Candidates
Look for any other areas where tasks are performed sequentially but could logically be run at the same time.

Example: A function that first downloads data and then loads a machine learning model. If these two tasks are independent, they could be run concurrently using threading or asyncio.

Report: Describe these opportunities, even if they were not the top bottlenecks in the profiling report, as they represent potential performance improvements.