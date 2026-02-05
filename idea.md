1\. Don’t move weights → activate fewer of them



Mixture-of-Experts



Gating



Conditional layers



→ Only load/compute part of the model per token. This is the core of what I was trying to achieve



Yes.

And this is the key clarification:



What you were building with streaming is not MoE.

It is weight residency control.



MoE / gating is a model-side decision.

Your DirectStorage idea is a runtime-side decision.



They solve different layers.



Concise mapping:



What you wanted



only touch the weights I actually need



Your mechanism:



stream / page weights



evict unused weights



That controls where weights live.



What MoE / gating does



only use some weights



That controls which weights participate in compute.



Why your approach alone cannot activate fewer weights



A dense transformer layer always needs:



all rows of W





to compute:



y = xW





So even if weights are streamed:

→ you still must read all of them.



No gating happens inside a dense layer.



The important connection



Your system becomes powerful when paired with MoE.



Because then:



gate decides expert IDs

→ runtime loads only those experts’ weights





That is the real alignment.



Short brutal truth



Streaming helps:

✔ memory

✔ load time

✔ residency



But it does not reduce compute.



MoE reduces:

✔ compute

✔ data movement

✔ bandwidth



This is the clean mental model



MoE decides what is needed



your runtime decides where it lives and when it is loaded



For your original goal



activate fewer of them



You cannot retrofit that into existing dense models.



You need:



an MoE model

or



conditional routing model



Your runtime is the missing infrastructure layer to make those usable locally.



One sentence:



You are building the machinery MoE models actually need to be practical on small machines.



1\. Don’t move weights → activate fewer of them



Mixture-of-Experts



Gating



Conditional layers



→ Only load/compute part of the model per token.



2\. Don’t store state → recompute it



Activation recomputation



Attention recompute



→ Trade FLOPs for memory.



3\. Don’t keep full history → window it



Sliding-window attention



KV eviction / rolling cache



→ KV stops growing forever.



4\. Don’t move full tensors → move blocks



Block-wise / tiled kernels



Small contiguous chunks



→ Better cache + paging + streaming.



(This matches your DirectStorage thinking.)



5\. Don’t move floats → move integers



Lower-bit quant



Pack aggressively



→ Bandwidth drops.



If I can solve all of these what will it allow me to do?



Short answer:



If you truly solve all five, you unlock a different operating point for local LLMs.



Here is exactly what that buys you.



✅ 1. You can run much larger models on small GPUs



Not marketing-large.

Practically:



7B → 30B+ class models on laptop GPUs

(and MoE models that look “70B+” on paper)



Because:



you don’t load most weights



you don’t keep most state



you don’t keep full history



✅ 2. You can run long context without exploding VRAM



Because:



KV is windowed



KV is evicted



some attention is recomputed



This removes the classic:



context length = death



problem.



✅ 3. You can make Ollama behave like a real runtime, not a loader



Meaning:



fast model switching



partial residency



dynamic loading



expert paging



background prefetch



This is huge for agents.



✅ 4. You can run MoE models properly on consumer hardware



This is the big one.



Without your 4 + 5:



MoE on small machines is mostly pointless.



With your system:



gate → load only chosen experts → compute → evict



That is exactly what you were intuitively aiming at.



✅ 5. You dramatically reduce the memory wall



The real constraint becomes:



bandwidth scheduling, not VRAM size.



That is a much better problem to have.



✅ 6. You make “70B on 8GB” almost true (but honestly)



Not dense 70B.



But:



70B MoE with 5–10B active parameters per token



—which is how the big labs actually scale now.



The one-line outcome



If you solve all five:



You turn local inference from a “load a model and pray” workflow into a paged, conditional, streaming compute system.



The honest boundary



Even with all five:



You still cannot escape:



total FLOPs needed for active parameters



But you do escape:



full model residency



full history



full tensor movement



In practical Ollama terms



You would enable:



➡ bigger models

➡ longer chats

➡ more agents

➡ less RAM

➡ less VRAM

➡ faster startup



on the same laptop.



That is not incremental.



That is a different class of local runtime.

