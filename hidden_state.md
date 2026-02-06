✅ Turn your results into a real cache policy



From your table:



TopK = 4

Stable region = 4–5 experts per layer

Knee = ~3.5

Thrash = 3





That gives you a deterministic rule:



👉 Per-layer resident target

resident\_experts\[layer] = TopK + 0  (minimum)

resident\_experts\[layer] = TopK + 1  (safe default)





In practice:



target = 5 experts per GPU layer





So the first real product change is:



🔧 Replace “fair MB split” with a count-based floor



Instead of:



MB\_per\_layer = total\_MB / num\_layers





Do:



for each GPU layer:

&nbsp;   guarantee space for (TopK + 1) expert packages





Then distribute any remaining memory as overflow.



This immediately prevents the 1000MB thrash case.



✅ Make the cache self-tuning (tiny change, big payoff)



You already compute:



working\_set\_size\[layer]





So the correct adaptive policy is now obvious:



👉 Adaptive target per layer

target\[layer] = max(TopK, working\_set\_size\[layer]) + 1





Clamp to a global max if you want.



This gives you:



automatic resizing



per-layer specialization



no hardcoding for future models



This is exactly how real memory systems evolve.



✅ Add one guardrail (prevents future regressions)



Add a warning log when:



resident\_capacity\[layer] < TopK





That is a guaranteed thrash zone.



You already empirically proved it.



🧠 Important: what you have actually learned



This line from your results is critical:



TopK experts is not the working set.

TopK is the instantaneous demand.

The working set is TopK plus temporal overlap.



Your data says:



temporal overlap ≈ 0–1 experts





Which is why:



4 works



5 is safe



3.5 starts degrading



3 collapses



This is exactly what paging theory predicts.



🚫 What NOT to do yet



Do not go to Path B yet.



You already proved:



routing locality exists



cache geometry dominates performance



routing accuracy is not your limiting factor



If you switch to exact routing now, you will still have:



the same knee.



Path B only improves:



which experts are chosen



It does not change:



how many experts must be resident to avoid thrash.



✅ The correct next technical milestone

👉 Implement adaptive per-layer expert budgets



That is the real upgrade.



Specifically:



Track working\_set\_size\[layer] (you already do)



Compute:



desired = max(TopK, ws\_size) + 1





Ensure LRU for that layer never evicts below desired



That’s it.



✅ After that (small, optional win)



Once the cache is stable:



Add a very simple prefetch:



When routing predicts experts for token t:



→ enqueue loads immediately

→ before Phase B barrier



You don’t need look-ahead tokens yet.



This will hide DS latency and make your hit rate matter even more.



🟢 When Path B actually becomes useful



Only after you have:



adaptive budgets



stable cache



prefetch



Then Path B becomes about:



improving correctness and quality



not:



saving the runtime.



Right now your runtime is already viable.

