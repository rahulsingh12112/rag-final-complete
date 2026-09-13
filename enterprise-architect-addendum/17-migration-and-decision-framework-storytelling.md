# Topic 17: Migration & Architect Decision Framework — Purane Se Naye Tak, Bina Sab Tod-e

> **Target Role:** AI Architect / Platform Architect
> **Ek line:** "Engineer batata hai 'system kya hoga'. Architect batata hai 'wahan **kaise pahunchoge** bina current business tod-e, aur kyun **ye** choice.' Yehi asli role hai."

---

## 🎬 Act 1 — CTO ka challenge (jo technical se zyada strategy tha)

Rahul ab confident tha. Uska RAG secure, resilient, observable, compliant sab tha. CTO ne bulaya:

*"Rahul, humara purana search system — 5 saal purana, keyword-based, 10,000 employees roz use karte hain — use tere naye RAG se replace karna hai. Par teen shart:*
1. *Ek din bhi down nahi ho sakta — log kaam kar rahe hain roz.*
2. *Agar RAG kharab nikla, wapas purane pe jaana **turant** aana chahiye.*
3. *Budget aur timeline realistic ho — mujhe board ko justify karna hai."*

Rahul ne socha: "RAG banana to aata hai. Par 10,000 log jispe depend karte hain, use **beech chalte-chalte** replace karna, bina kisi ko disturb kiye, safety-net ke saath... ye alag khel hai."

Ye **migration** ka problem hai — aur ye woh cheez hai jo engineer aksar nahi sochta par architect ke liye **roz ka kaam** hai. Kyunki asli duniya me tum kabhi khaali maidan me system nahi banate — hamesha ek **chalta hua purana system** hota hai jise tod-na nahi hai.

Dost ne kaha ek line jo poore topic ka saar hai:
> *"Naya system banana 20% kaam hai. Purane se naye pe **safely shift** karna 80% kaam hai. Aur bade organizations isi 80% ke liye architect ko paisा dete hain."*

---

## 🧩 Act 2 — Migration ka pehla rule: "Big Bang mat karo"

Rahul ka pehla instinct: "Ek weekend, purana band, naya chालu. Ho gaya."

Dost ne sar pakda. Ye **"Big Bang" migration** hai — sabse tempting, sabse khatarnaak.

```
BIG BANG:  Purana OFF ──────► Naya ON  (ek jhatke me)
           ▲
           💣 Agar naya fail hua? 10,000 log stuck. Wapas jaana? Mushkil.
              Weekend bhar debug. Monday ko aag. Career risk.
```

**Enterprise rule: Migration hamesha INCREMENTAL hoti hai** — thoda-thoda, dekh-dekh ke, wapas-jaane ka raasta khula rakh ke. Kyun? Kyunki "safely" ka matlab hi ye hai: har kadam pe **agar galat hua to peeche hat sakein.**

---

## 🪜 Act 3 — Do bhai patterns: Strangler Fig + Parallel Run

Dost ne do classic patterns samjhaaye. Naam ajeeb hain par idea sona hai.

**Pattern 1 — Strangler Fig (dheere-dheere purane ko replace karo):**

Naam ek ped se aaya — strangler fig ek purane ped ke charon taraf ugta hai, dheere-dheere use replace karta hai, jab tak naya ped khud khada na ho jaaye. Purana ped kabhi ek jhatke me nahi girta.

```
Phase 1:  User → [Router] → 100% purana system
Phase 2:  User → [Router] → 90% purana, 10% naya RAG  (ek chhota feature RAG se)
Phase 3:  User → [Router] → 50% purana, 50% naya
Phase 4:  User → [Router] → 100% naya RAG
          (ab purana band karo — "strangle" ho gaya)
```

Ek **router** (API gateway / feature flag) beech me baithta hai jo decide karta hai kaunsi request purane ko jaaye, kaunsi naye ko. Dheere-dheere naye ka % badhao. Kabhi bhi problem → router se wapas purane ko bhej do.

**Pattern 2 — Parallel Run / Shadow Mode (dono chalao, compare karo):**

Ye aur bhi safe hai. Naye system ko live traffic do **par uska jawab user ko mat dikhao** — sirf background me chalao aur purane se compare karo.

```
User → Purana system → jawab USER ko (ye dikhta hai)
         │
         └──(same query bhi)──► Naya RAG → jawab LOG karo (user ko NAHI)
                                            │
                              Compare: naya == purana? behtar? kharab?
                              Confidence banao BINA user ko risk diye.
```

Hafton "shadow mode" me chalao. Data ikattha karo — naya RAG kitni baar sahi, kitni baar galat, latency kaisी. Jab confidence aa jaaye, **tab** actual traffic dena shuru karo (Strangler ki tarah). User ko poore process me pata bhi nahi chalta.

> **Interview me killer combo:** *"Main pehle **shadow mode** me naye system ko live traffic dikhata hoon bina user ko jawab diye — real data pe confidence banane ke liye. Phir **strangler fig** se dheere-dheere traffic shift karta hoon, feature-flag/router se, taaki kisi bhi kadam pe instantly rollback kar sakun. Big-bang kabhi nahi — wapas jaane ka raasta hamesha khula."*

---

## 🪜 Act 4 — Rollback: safety-net jo har kadam pe ho

CTO ki doosri shart thi: "wapas jaana turant." Ye **rollback** hai — aur incremental migration ka sabse bada fayda yehi hai.

```
Har phase pe:
  Naya % badhaya → monitor (SLO se — Topic 15!) → theek? aage badho
                                                 → kharab? router wapas purane pe (seconds)

Data ka dhyaan: agar naya system data likh raha hai, rollback pe wo data purane
                system ke saath consistent hona chahiye. Isliye migration me
                aksar "dual-write" ya "purana source-of-truth rahe" rakhte hain.
```

Rollback ko monitoring (Topic 15) se jodo: SLO breach hote hi automatic ya one-click wapas. Rahul ne dekha — pichhle saare topics **yahan aa ke jud rahe the.** Migration akela nahi — wo secure (13), resilient (14), observable (15), governed (16) system ko **safely shift** karne ki kala hai.

---

## 🎬 Act 5 — Ab asli architect wali cheez: DECISIONS

Migration plan ban gaya. Par CTO ne teesri shart di thi: "board ko justify karo." Aur board technical nahi hai — unhe "Qdrant vs Pinecone" nahi samajh aata. Unhe **decisions aur unke reasons** chahiye. Yahan Rahul ko engineer se architect banna tha.

Dost ne kaha: "Architect interview ka 50% yehi hai — **tumhe pata hona chahiye kaise decide karte hain, na ki sirf kya banate hain.** Chal tujhe 3 decision frameworks deta hoon jo har interview me kaam aayenge."

---

## 🧩 Act 6 — Framework 1: Build vs Buy (khud banau ya khareedu?)

Har architect ke saamne ye sawaal aata hai: ye component khud banayein ya ready-made le lein? (e.g. self-hosted Qdrant vs managed Pinecone, khud LLM host vs OpenAI API)

```
┌─────────────────────────────────────────────────────────────┐
│  BUY (managed/ready) jab:                                     │
│    • Ye tumhara "core differentiator" NAHI hai               │
│    • Team chhoti, time kam                                    │
│    • Scale abhi chhota (managed sasta padta hai chhote pe)    │
│    • "undifferentiated heavy lifting" hai (sab karte hain)    │
│                                                               │
│  BUILD (khud) jab:                                            │
│    • Ye tumhari core value hai (ispe compete karte ho)        │
│    • Scale bada (managed mehnga ho jaata hai — Topic 3 dekha) │
│    • Special control chahiye (compliance, latency, tuning)    │
│    • Vendor lock-in ka risk zyada hai                         │
└─────────────────────────────────────────────────────────────┘
```

**RAG example:** LLM khud host karna? Zyादातर mat karo — OpenAI/Bedrock **buy** karo (LLM banana tumhara core nahi, aur GPU chalाna narak hai). Vector DB? Chhote pe Pinecone **buy**, bade pe self-host Qdrant **build** (Topic 3 ka breakeven — 5M vectors). 

> **Golden line:** *"Main un cheezon ko buy karta hoon jo 'undifferentiated heavy lifting' hain — jo sab ko chahiye aur jo mera competitive advantage nahi — aur build sirf apne core differentiator pe karta hoon. Har build ki ek chhupi cost hai: maintenance, on-call, upgrades. 'Build' ka matlab 'humesha ke liye own karna' hai."*

---

## 🧩 Act 7 — Framework 2: Trade-off sochne ka tareeka (koi "best" nahi hota)

Sabse bada architect signal: **jaan lo ki "perfect" ya "best" answer hota hi nahi — sirf trade-offs hote hain.** Junior bolta hai "X best hai." Architect bolta hai "X ye deta hai par ye kho deta hai; hamare case me ye trade-off theek hai kyunki..."

Har architecture decision teen cheezon ke beech khींchtan hai (aur aksar ek "AI-era" chautha):

```
        COST 💰
         /  \
        /    \
  SPEED ⚡───── RELIABILITY 🛡️
   (latency)   (uptime/HA)

  + RAG me chautha kона: QUALITY/ACCURACY 🎯
    (fast+sasta+reliable par galat jawab = bekaar)

  Kisi ek ko upar karo → doosra neeche aata hai.
  "Sab-kuch max" impossible. Architect decide karta hai
   is business ke liye kaun-sा kona zyada matter karta hai.
```

**RAG example bol ke dikhao:**
- "Latency chahiye? → semantic cache + chhota model → par quality thodी kam ho sakti (trade-off)."
- "Quality chahiye? → bada model + reranking → par latency aur cost badhega."
- "Cost bachao? → cheaper model routing → par edge cases pe quality gir sakti."

> **Interview me hamesha ye structure use karo:** *"Ye choice X deta hai, par Y ki keemat pe. Hamare case me [business context] ke hisaab se [ye] zyada matter karta hai, isliye main [choice] karunga — aur [risk] ko [mitigation] se sambhalunga."* Ye ek line tumhe har baaki candidate se alag kar degi.

---

## 🧩 Act 8 — Framework 3: Napkin Math (live estimation — tumhari weak spot!)

> ⚠️ Rahul, ye tumhari known weak-spot #4 hai (estimation — 166 vs 17 slip, LLM ~10-20 req/s NOT 1000). Isliye ye section extra dhyaan se. Interview me ye **live, bolte-bolte** karna padta hai. Practice karo.

Architect se board/interviewer poochhega: "10,000 employees ke liye ye system kitna bada, kitna mehnga?" Tumhe **napkin pe, seconds me**, roughly sahi number nikalna hai. Perfect nahi — **roughly sahi, confidently.**

**Method — hamesha yehi 4 step:**
```
1. Users → requests: kitne log, kitni baar? 
   10,000 employees × 20 queries/day = 200,000 queries/day
   
2. Per-day → per-second (yaad rakho: 1 din ≈ 86,400 sec ≈ 100,000 sec round karo):
   200,000 / 100,000 = 2 queries/sec average
   Peak = average × 3 se 5 = ~10 queries/sec peak
   
3. Capacity: ek RAG replica kitne sambhal sakti?
   LLM-bound system ~10-20 req/s per replica (NOT 1000! ye tumhari weak spot)
   → 10 req/s peak = 1-2 replicas kaafi (+ buffer = 3 for HA)
   
4. Cost: LLM calls sabse bada. 200k queries/day × ~4k tokens × price...
   (Topic 9 ka cost table yaad — 10k q/day ≈ $1,640/mo; 200k ≈ ~$5-8k/mo range)
```

**Napkin math ke rules (rat lo):**
```
• 1 din ≈ 86,400 sec → round to 100,000 for easy math
• Peak = average × 3-5 (kabhi flat mat maano)
• LLM system: 10-20 req/s per replica (GPU/API bound) — NOT thousands
• 70B model ≈ 140GB VRAM ≈ multi-GPU (1 GPU me nahi aata)
• Buffer laga ke bolo: "roughly 3 replicas, main 4 rakhunga safety ke liye"
• Divide DHYAAN se — ek zero idhar-udhar = 10x galat (tumhari 166 vs 17 wali galti)
```

> **Interview me:** dheere bolo, steps loud bolo ("200k per day, divide by ~100k seconds, that's ~2 per second average, peak maybe 10..."). Interviewer **process** dekhta hai, exact number nahi. Confident round numbers > precise ghabraya hua. Galti ho jaaye to "let me sanity-check that" bol ke wapas dekho — ye maturity dikhata hai.

---

## 🧩 Act 9 — Framework 4: Stakeholders (architect = translator)

Ek aur cheez jo engineer nahi sochta: architect ko **alag-alag logon se alag bhaasha** me baat karni hoti hai. Same RAG system, alag audience:

```
CEO/Board ko:      "Ye system employees ki productivity 20% badhaega,
                    cost $8k/month, 3 mahine me live. ROI 6 mahine me."
                    (business value, cost, timeline — NO tech jargon)

Security team ko:  "IRSA, encryption at-rest, audit logs, PII redaction,
                    SOC2 compliant." (Topic 13, 16)

Dev team ko:       "FastAPI orchestrator, Qdrant, OTel tracing, GitOps deploy."
                    (implementation detail)

Finance ko:        "Per-query cost $0.005, scales down at night, spot instances
                    for batch." (FinOps)
```

> **Architect ki asli super-power:** **translation.** Tum tech aur business ke beech pul ho. Interview me agar tum business language bhi bol sako ("ROI", "risk", "timeline", "productivity") — sirf tech nahi — to tum **senior architect** dikhte ho. Ye woh cheez hai jo staff/principal level pe expect hoti hai.

---

## 🛡️ Act 10 — Poora migration + decision framework (ek jagah)

```
┌──────────────────────────────────────────────────────────────────────┐
│              MIGRATION (purane → naye, safely)                         │
│                                                                        │
│   1. Shadow Mode   → naya live traffic pe, jawab user ko NAHI, compare │
│   2. Strangler Fig → router se dheere % shift (10→50→100)              │
│   3. Rollback      → har phase pe SLO monitor, kharab → router wapas   │
│   4. Data safety   → dual-write / purana source-of-truth till confident│
│   5. Cleanup       → 100% confident → purana band ("strangled")        │
│                                                                        │
├──────────────────────────────────────────────────────────────────────┤
│              ARCHITECT DECISION FRAMEWORKS                             │
│                                                                        │
│   • Build vs Buy    → core differentiator? scale? lock-in?             │
│   • Trade-offs      → "best" nahi hota; cost/speed/reliability/quality │
│   • Napkin Math     → users→req/s→replicas→cost (confidently rough)    │
│   • Stakeholders    → CEO/security/dev/finance ko alag bhaasha         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Act 11 — Interview Ready

### 🟢 30-second Architect Answer (Migration)

> "Main kabhi big-bang migration nahi karta. Pehle **shadow mode** — naye system ko live traffic dikhata hoon bina user ko jawab diye, aur purane se compare karke confidence banata hoon. Phir **strangler fig** pattern — ek router/feature-flag se dheere-dheere traffic shift karta hoon (10% → 50% → 100%), aur har phase pe SLO monitor karta hoon taaki kharab hote hi **instant rollback** ho sake. Purana system source-of-truth rehta hai jab tak main 100% confident na ho jaaun. Ye teen shart poori karta hai: zero downtime, easy rollback, aur measured risk."

### 🟢 30-second Architect Answer (Decisions)

> "Har decision me main pehle poochhta hoon — ye core differentiator hai ya undifferentiated heavy lifting? Uspe build-vs-buy decide hota hai. Phir main 'best' nahi, **trade-offs** me sochta hoon — har choice cost, speed, reliability, ya quality me se kisi ki keemat pe aati hai, aur main business context ke hisaab se decide karta hoon kaunsा matter karta hai. Estimation main roughly-right, confidently karta hoon. Aur main har stakeholder — board, security, dev — ko unki bhaasha me explain karta hoon."

### 🔵 5-min version

Upar dono + shadow/strangler diagrams + rollback+data-consistency + build-vs-buy (undifferentiated heavy lifting line) + trade-off triangle+quality corner + napkin math 4-step live + stakeholder translation.

---

## ❓ Act 12 — Follow-ups

**Q: "Migration ke beech me data model change ho to (purana schema ≠ naya)?"**
> A: **Anti-corruption layer** — ek translation layer jo purane aur naye ke beech data convert kare, taaki dono apne format me kaam karte rahein. Migration ke baad hata do. Isse ek system ki galti doosre ko corrupt nahi karti.

**Q: "Shadow mode me naya system galat jawab de raha hai — kaise pata jab user ko dikha hi nahi rahe?"**
> A: Automated comparison + sampling. Naye aur purane ke jawab ko ek judge (LLM-as-judge ya rules ya human sample) se compare karo. Metrics: agreement rate, jahan differ kiya wahan kaun sahi. RAG me groundedness/relevance (Topic 15) bhi shadow me measure karo. Confidence numbers se banao, gut se nahi.

**Q: "Board bole 'itna slow kyun? Big bang karo, jaldi'?"**
> A: Risk-cost me samjhao (business language): "Big bang me agar fail hua to 10,000 log × 1 din down = $X loss + trust. Incremental me har kadam safe, extra 3 hafte lagenge par risk near-zero. 3 hafte vs potential $X — ye insurance hai." Board ko **numbers aur risk** me convince karo, technical purity me nahi.

**Q: "Ye purana system tumne banaya nahi — legacy hai, samajh nahi aa raha. Migrate kaise?"**
> A: Pehle **characterize** karo — shadow/logging se samjho purana actually karta kya hai (aksar documentation galat hoti hai, behavior real hota hai). Uske real behavior ko "spec" maano. Strangler se ek-ek feature replace karo, har ek ko naye ke against verify karo. Poora legacy ek saath samajhne ki koshish mat karo — piece by piece.

**Q: "Architect aur senior engineer me farq ek line me?"**
> A: "Senior engineer best solution banata hai. Architect best solution choose karta hai given constraints — cost, team, timeline, risk, stakeholders — aur us tak safely pahunchne ka raasta banata hai. Engineer 'kya' me expert, architect 'kya + kyun + kaise-pahunche + kis keemat pe' me."

---

## 🏁 Act 13 — Kahani ka Anth (aur tumhara start)

6 mahine baad. Rahul ka RAG ab poori company chala rahi thi. Purana system chup-chaap "strangle" ho chuka tha — kisi ko pata bhi nahi chala kab shift hua. Zero downtime. CTO khush. Board ne budget approve kiya.

Rahul ab **architect** tha. Woh insaan jo us Friday raat $40,000 ka bill laya tha, ab **decisions** leta tha — build vs buy, trade-offs, migration strategy — aur unhe board ke saamne defend karta tha.

Farq kya tha? Usne code likhna better nahi kiya. Usne **sochna** better kiya. Har cheez ko poochna seekha: *"kyun, kab fail, kaise evolve, kitna kharcha, kaun approve."*

> **Ye poora addendum yehi sikhata hai** — 5 kahaniyan, ek journey: secret leak (13) → region down (14) → slow mystery (15) → salary leak (16) → migration (17). Har disaster ne Rahul ko engineer se architect banaya.

**Ab tumhari baari.** Interview me tum Rahul ho — par ab tum galtiyan pehle hi jaante ho. Jab interviewer disaster ka scenario de ("system down ho gaya to?", "data leak ho gaya to?", "purane se migrate kaise?"), tumhare paas sirf jawab nahi — **kahani** hai. Aur kahani kabhi nahi bhoolti. 🎬

---

## 📌 Ek panne ka summary

| Concept | Matlab | Trick |
|---------|--------|-------|
| Big Bang | Ek jhatke me switch | ❌ kabhi nahi (career risk) |
| Shadow Mode | Naya live, jawab user ko nahi | Confidence bina risk |
| Strangler Fig | Router se dheere % shift | Purana dheere "strangle" |
| Rollback | Kharab → wapas (SLO se) | Har kadam safety-net |
| Anti-corruption layer | Purana↔naya translator | Ek doosre ko corrupt na kare |
| Build vs Buy | Core? scale? lock-in? | Undifferentiated heavy lifting = buy |
| Trade-offs | "Best" nahi, keemat | Cost/Speed/Reliability/Quality |
| Napkin Math | users→req/s→replicas→cost | 1 din≈100k sec, peak×3-5, LLM 10-20 r/s |
| Stakeholders | Alag bhaasha alag logon ko | Architect = translator |

**Core mantra:** *Big-bang kabhi nahi — shadow → strangler → rollback ready. "Best" nahi hota, trade-offs hote hain. Roughly-right confidently. Har stakeholder ko uski bhaasha. Architect = kya+kyun+kaise-pahunche+kis-keemat-pe.*

---

## 🎓 Poore Addendum ka Nichod (5 topics, ek nazar)

| # | Topic | Ek line jo yaad rakhni hai |
|---|-------|---------------------------|
| 13 | Security & Secrets | Zero-trust; secret git me nahi; IRSA identity; mTLS |
| 14 | Multi-Region HA/DR | RTO/RPO business decide kare; Multi-AZ default, Multi-Region zaroorat pe; test karo |
| 15 | Tracing + SLO | Logs+Metrics+Traces; Trace ID jodta hai; SLO strict SLA dheela; error budget; RAG me quality bhi naapo |
| 16 | Data Governance/PII | Access control retrieval pe (prompt pe nahi); classify+redact+audit+encrypt; GDPR me vectors bhi delete |
| 17 | Migration + Decisions | Shadow→strangler→rollback; build-vs-buy; trade-offs; napkin math; stakeholder translation |

**Tum ab AI/Platform Architect interview ke liye ~90%+ ready ho. Baaki 10% = live mock interviews + ye kahaniyan bol-bol ke practice.** 🚀
