\# Dense → Block-MoE (Structural Experts) — Exact Equivalence \& Runtime Paging



This note documents a \*\*lossless way to split a dense FFN layer into independent blocks (“experts”)\*\* so the model can be \*\*paged / streamed / scheduled\*\* like MoE — while remaining \*\*mathematically identical\*\* to the original dense model \*when all blocks are executed\*.



This is intended for a systems runtime that:

\- pages weights on demand,

\- supports expert/block residency,

\- and may later choose to execute only a subset of blocks under budget.



---



\## Core Idea



A dense FFN layer can be rewritten as a \*\*sum of independent sub-FFNs\*\* by slicing the large hidden dimension.



This is not a new model.

It is a new \*layout and execution strategy\*.



---



\## Original Dense FFN



For one transformer layer:



h = act(x · W\_up) · W\_down





Shapes:



x : \[B, d\_model]

W\_up : \[d\_model, d\_ff]

W\_down : \[d\_ff, d\_model]





---



\## Structural Split (Block Experts)



Split the wide dimension `d\_ff` into K blocks:



W\_up = \[ W\_up\_0 | W\_up\_1 | ... | W\_up\_{K-1} ]

W\_down = \[ W\_down\_0

W\_down\_1

...

W\_down\_{K-1} ]





Where:



W\_up\_i : \[d\_model, d\_ff\_i]

W\_down\_i : \[d\_ff\_i, d\_model]





---



\## Execution Form



Instead of one big FFN, compute:



h\_i = act(x · W\_up\_i) · W\_down\_i

h = sum\_i h\_i





---



\## Exact Equivalence Guarantee



If \*\*all blocks are executed\*\*:



sum\_i ( act(x · W\_up\_i) · W\_down\_i )





is \*\*mathematically identical\*\* to:



act(x · W\_up) · W\_down





This is a pure linear-algebra identity.



There is:

\- no approximation

\- no retraining

\- no modeling change



Only the execution graph changes.



Minor floating-point differences may occur due to different accumulation order (normal GPU behavior).



---



\## Critical Pairing Rule



To preserve equivalence:



\- split `W\_up` on \*\*columns\*\*

\- split `W\_down` on \*\*rows\*\*



and pair block `i` with block `i`.



This exact pairing is required:



act(x·W\_up\_i) → W\_down\_i → sum





Do NOT:

\- concatenate activations and apply a single W\_down

\- or mix block boundaries



---



\## Why This Is Perfect for Paging / Streaming Runtimes



Each block:



(W\_up\_i, W\_down\_i)





becomes:



\- an independent unit of compute

\- an independent unit of residency

\- an independent unit of IO



So the FFN becomes:



sum of pageable tiles





instead of:



one monolithic matrix





---



\## Dense Behavior vs Budgeted Behavior



\### Dense mode



Execute all blocks:



h = sum\_i h\_i





→ identical outputs  

→ identical logits  

→ identical model



\### Budgeted / MoE-like mode



Execute only a subset S:



h ≈ sum\_{i ∈ S} h\_i





This is now:



\- a truncated dense layer

\- intentionally approximate

\- quality degrades, usually smoothly



This allows:



\- dynamic memory budgets

\- runtime density control

\- demand-paged execution



---



\## Important: This Is NOT a Traditional MoE



There is no semantic expert specialization.



These “experts” are:



\- structural slices of a dense layer



The router is therefore not semantic — it becomes a \*\*scheduler\*\*:



policy(x, layer, token, budget) → which blocks to run





---



\## Best Split Axis



Always split on the large hidden dimension:



W\_up : \[d\_model, d\_ff] ← split on d\_ff

W\_down: \[d\_ff, d\_model]





This avoids:



\- partial dot-product reductions

\- cross-block reductions

\- complicated synchronization



This makes each block:



x → FFN block → full d\_model output





---



\## Where to Apply This in Practice



This applies primarily to:



\- MLP / FFN layers



because they hold the majority of parameters.



You do NOT need to touch:



\- attention projections

\- QKV

\- output heads



to get most of the memory and paging benefit.



---



\## Export / Layout Strategy



During export (e.g. GGUF):



For each FFN layer:



ffn\_up.weight → expert\_0.up, expert\_1.up, ...

ffn\_down.weight → expert\_0.down, expert\_1.down, ...





Each expert entry maps directly to:



(file\_offset, size)





which is ideal for block-level streaming.



---



\## Key Runtime Insight



This turns:



Dense model execution



into:



a sum of independently schedulable compute tiles





This is equivalent to turning a monolithic memory object into

a set of virtual pages.



---



\## One-Line Summary



Splitting a dense FFN into block experts is:



✔ lossless if all blocks are executed  

✔ perfectly compatible with streaming and VMM  

✔ a clean bridge between dense models and MoE-style scheduling  



This is effectively:



\*\*demand-paged dense inference.\*\*

