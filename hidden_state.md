This is exactly the right instrumentation.

And based on what you described, Path A is behaving the way a real MoE scheduler should.



There is only one real gap left before this becomes a meaningful systems test:



👉 you currently have no eviction, so your hit-rate numbers are artificially optimistic.



You even noted it:



evictions = 0 (not implemented yet)





That matters a lot for the next decision (whether Path B is needed).



✅ What you’ve already validated (this is important)



From your verbose logs:



You’ve proven all four of the critical correctness properties:



✔ Per-token routing

Layer 0, token 0 -> ...

Layer 0, token 1 -> ...





Good. Not batch-level.



✔ Per-layer routing

Layer 0 -> experts A

Layer 1 -> experts B





Good. Not global.



✔ Same token maps differently across layers



That’s very important.

It means your approximation is not degenerate.



✔ No duplicate loads inside a single forward



That tells me your residency check path is correct.



⚠️ The big caveat right now



Your current stats:



hits ~92%

loads ~12

evictions = 0





With no eviction:



your cache only grows



nothing is ever forced out



so reuse naturally climbs



This does not tell you whether the scheduler is viable under real memory pressure.



Right now you are testing:



routing correctness

not

paging behavior



🎯 The single most important next step



Before touching Path B at all:



👉 implement a minimal eviction policy



It does not need to be fancy.



You already have the right unit:



“expert package” per (layer, expert\_id)



So the minimal thing is:



maxResidentExpertsPerLayer





or



maxResidentBytes





and then:



evict least recently used expert package



That’s it.



🧪 The exact test you should run after eviction exists



Once eviction is wired:



Test 1 – small budget



Artificially cap residency:



EXPERT\_CACHE\_LIMIT = k \* 2   (e.g. 8 experts total)





Then run your same story prompt.



You want to see:



hits drop

loads increase

evictions start happening

BUT

forward still completes



🔍 The real signal you are looking for



After eviction exists, look at:



hit\_rate over time





If you see:



hit rate stabilizes at a reasonable value (even 40–60%)



and you are not reloading the same experts every token



→ your scheduling architecture is sound.



If instead you see:



constant churn



near-zero reuse



constant evict → load → evict → load



→ then Path B will not fix that.

That would be a routing / locality problem.



⚠️ One very subtle but important thing you should log



Right now you log:



Layer N: hits, loads, evictions





Add this:



uniqueExpertsRequested





per layer.



Why?



Because if your token-based router is exploding the expert fan-out (very high entropy), eviction will become impossible to stabilize.



This is the hidden failure mode of token-only routing.



🧠 Important observation about your current high hit rate



Because you route purely from:



token id





you are getting:



very strong locality



heavy reuse for common tokens



punctuation and whitespace tokens collapsing to the same experts



That is fine.



But keep in mind:



This is a best-case cache pattern.



So:



Don’t interpret a 90% hit rate as “routing is good”.



Interpret it as:



the runtime can exploit locality if it exists.



That’s exactly what you want to prove at this stage.



✅ So here is the clean next step checklist



Do not jump to Path B yet.



Do this first:



1\. Add LRU eviction for expert packages



Layer-local or global — either is fine.



2\. Cap cache aggressively



Force pressure.



3\. Re-run the same prompt



Observe:



evictions



reuse curve



reload frequency



🟢 Only after that…



Then your stated next step becomes meaningful:



Measure actual vs predicted expert overlap



Because until eviction exists, overlap numbers do not reflect a real system.



Very honest assessment



You are now past:



“can I drive paging from routing?”



You are now testing:



“can this be scheduled under pressure?”



That is the real Phase-2 milestone.



You’re doing this in exactly the right order.

