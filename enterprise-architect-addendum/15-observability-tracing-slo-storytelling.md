# Topic 15: Distributed Tracing + SLO — 6 Services, Gayab Kahan Hua Pata Hi Nahi

> **Target Role:** AI Architect / Platform Architect
> **Ek line:** "Logs batate hain 'kya hua', metrics batate hain 'kitna hua', tracing batata hai 'kahan hua'. Teeno chahiye. Aur SLO batata hai 'acceptable kya hai'."

---

## 🎬 Act 1 — "Down nahi hai, bas... slow hai" (sabse bura wala problem)

Rahul ka RAG system multi-region ho chuka tha, secure tha. Sab badhiya. Phir ek din ek customer ne complain kiya: "Bhai, jawab aata hai, par bahut slow. 8-9 second lag rahe hain."

Rahul ne khud try kiya. Sach me slow. Par... **down bhi nahi tha.** Health check green. CPU normal. Koi error nahi. Bas... slow.

Ye down se bhi bura problem hai. Down hota to saaf pata — kuch toota. Par "slow" me kya toota? Kuch bhi nahi toota. Kuch **dheere** ho raha hai. Par **kaun**?

Rahul ke paas 6 services thi:
```
Orchestrator → Guardrails → Embedding → Retrieval → LLM Gateway → (Guardrails output)
```

Request in sabse guzarti thi. Slow kaunsi thi? Rahul ne har service ke logs alag-alag khole:

```
[Orchestrator log]  "9:42:01 request aayi, 9:42:09 jawab bheja"     ← 8 second lage
[Guardrails log]    "9:42:01 check kiya, ok"                         ← kab? kitna?
[Embedding log]     "9:42:0? embedded"                              ← kaunsi request thi?
[Retrieval log]     "9:42:0? searched, 5 results"                   ← kitna time?
[LLM log]           "9:42:0? generated 512 tokens"                  ← ye slow tha kya?
```

**Problem:** Har service apni alag diary likh rahi thi. Kisi ko nahi pata ki "9:42:01 wali orchestrator request" **kaunsi** embedding call thi, **kaunsi** retrieval thi. 6 alag diaries, aur unhe **jodne ka koi tareeka nahi**. Ek hi second me 50 requests aa rahi thi — kaunsi log line kis request ki hai? 🤯

Rahul 2 ghante logs me dooba raha. Pata nahi chala. Haar kar dost ko call kiya.

Dost: "Tune **distributed tracing** lagaya hai?"
Rahul: "Nahi. Woh kya hai?"
Dost: "Yehi teri problem hai. Chal samjhaata hoon."

---

## 🧩 Act 2 — Teen dost: Logs, Metrics, Traces (Observability ke 3 pillar)

Dost ne kaha: "System ko 'dekhne' ke 3 tareeke hain. Log inhe alag samajhte nahi, isliye phaste hain. Suno:"

```
┌─────────────────────────────────────────────────────────────────┐
│  📜 LOGS     = "KYA hua"                                          │
│     Ek-ek event ki diary. "Error aaya", "user login hua".        │
│     Detail bahut, par ek request ka poora safar nahi dikhta.     │
│                                                                   │
│  📊 METRICS  = "KITNA hua" (numbers over time)                   │
│     "abhi 500 req/sec", "p99 latency 8s", "CPU 60%".             │
│     Trend dikhta hai, par WHY nahi. "Slow hai" pata chalega,     │
│     "kaunsi service slow" nahi.                                   │
│                                                                   │
│  🔍 TRACES   = "KAHAN hua" (ek request ka poora safar)           │
│     Ek request 6 services me kahan-kahan gayi, har jagah          │
│     kitna time laga — ek hi timeline pe. Rahul ki asli zaroorat. │
└─────────────────────────────────────────────────────────────────┘
```

**Analogy — courier package tracking:**
- **Log** = har warehouse ki apni register ("aaj 500 packets aaye")
- **Metric** = "aaj average delivery 3 din" (number, trend)
- **Trace** = **tumhare** ek packet ka live tracking — "Delhi warehouse: 2 din atkа 😤, phir Mumbai: 1 ghanta, phir delivered." **Ek specific journey, har stop ka time.**

Rahul ko yehi chahiye tha — ek slow request ka poora journey, har service pe kitna ruki.

---

## 🪜 Act 3 — Tracing ka jaadू: "Trace ID" (ek dhaaga jo sabko jodta hai)

Distributed tracing ka core idea itna simple hai ki Rahul ko gussa aaya ki pehle kyun nahi socha.

**Idea:** Jab request pehli baar aaye (orchestrator pe), use ek **unique ID do — "Trace ID"**. Phir wo request jis-jis service me jaaye, **ye ID saath le jaaye**. Har service apna kaam karke bole "maine Trace-ID `abc123` pe itna time laga."

Ab sab logs ko us ID se **jod** sakte ho:

```
Trace ID: abc123  (ek request ka poora safar, ek timeline pe)

Orchestrator  ├████████████████████████████████████████┤  8000ms (total)
  Guardrails  ├█┤ 50ms
  Embedding   ├█┤ 30ms
  Retrieval   ├██┤ 100ms
  LLM Gateway            ├████████████████████████████┤  7500ms  ← 😱 YEH RAHA MUJRIM!
  Guard(out)                                          ├█┤ 50ms
```

Ek nazar me saaf: **LLM Gateway 7500ms le raha tha!** Baaki sab fast. Rahul ne 2 ghante logs me jo nahi dhoondh paya, wo trace timeline me **2 second** me dikh gaya.

**Vocabulary (interview me bolna aata hai):**
- **Trace** = poora safar (ek request end-to-end)
- **Span** = safar ka ek hissa (ek service ka kaam) — har service ek span banati hai
- **Trace ID** = poore safar ki common ID
- **Span ID** = har hisse ki apni ID (aur "parent span ID" batati hai kaun kiske andar hai)
- **Context propagation** = Trace ID ko ek service se agli service tak le jaana (aksar HTTP header me — `traceparent`)

> **Ye "context propagation" hi asli kaam hai.** Agar orchestrator retrieval ko call kare par Trace ID pass na kare, to trace toot jaata hai — retrieval ka span kisi aur galaxy me chala jaata hai. Isliye har service ko ID **aage bhejni** padti hai.

---

## 🪜 Act 4 — Ye khud likhna? Nahi. OpenTelemetry aur Jaeger

Rahul: "To har service me main ID generate karun, header me daalun, aage bhejun, time measure karun... itna code?"

Dost: "Nahi re. Iske liye **OpenTelemetry (OTel)** hai — ek standard library. Wo 90% kaam khud karti hai. Tu bas thoda setup karta hai, wo har request pe apne aap span banati hai, ID propagate karti hai."

**Do parts:**
1. **OpenTelemetry** = **collect** karta hai (har service me, spans banata + ID propagate karta)
2. **Jaeger** (ya Tempo, Zipkin) = **dekhne** ka dashboard — wo timeline picture jo upar dikhi

```
Har service  ──(OTel spans + trace ID)──►  Collector  ──►  Jaeger UI
  (auto-instrumented)                                      (timeline dikhaता)
```

**FastAPI me actual code (Rahul ke orchestrator me):**

```python
# Setup — ek baar, app start pe
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger-collector:4317"))
)

# Ye do line JAADU hai — FastAPI aur outgoing requests ko auto-instrument
FastAPIInstrumentor.instrument_app(app)   # har incoming request pe span
RequestsInstrumentor().instrument()       # har outgoing call pe ID auto-propagate
```

Bas. Ab **har** request pe span apne aap banega, aur jab orchestrator `requests.post()` se retrieval ko call karega, Trace ID **automatically** header me chala jaayega. Rahul ko manual kuch nahi karna.

**Custom span** (jab tum apna specific hissa naapna chaho):

```python
tracer = trace.get_tracer(__name__)

def process(self, query):
    with tracer.start_as_current_span("retrieval_call") as span:
        span.set_attribute("query.length", len(query))
        span.set_attribute("top_k", 5)
        chunks = self.retrieval.search(query)      # ye call apne aap child span banegi
        span.set_attribute("chunks.found", len(chunks))
        return chunks
```

Ab Jaeger me tumhe "retrieval_call" span dikhega, uske andar attributes (query length, chunks found) — debugging ke liye sona.

---

## 🎬 Act 5 — Rahul ne mujrim pakda, par ek naya sawaal (SLO ki entry)

Trace se pata chala LLM Gateway slow tha. Rahul ne dekha — ek naya bada model laga diya tha jo dheere generate karta tha. Fix kiya, latency wapas 2s.

Par dost ne ek gehra sawaal poochha: "Ruk. Tujhe pata kaise chala ki 8 second 'slow' hai aur 2 second 'theek'? Kisne decide kiya? Tere gut feeling ne? Ya koi likhit target hai?"

Rahul chup. Uske paas koi **definition** nahi thi ki "acceptable" kya hai. Jab customer chillaya, tab pata chala. **Reactive** tha, proactive nahi.

Dost: "Yahi **SLO** solve karta hai. Chal ye teen shabd samajh — SLI, SLO, SLA. Log inhe sabse zyada ulat-pulat karte hain."

---

## 🧩 Act 6 — SLI, SLO, SLA (teen bhai, unko kabhi confuse mat karna)

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│  SLI (Indicator)  = MAAP — actual number, abhi kya hai            │
│      "p99 latency abhi 2.1s hai", "success rate 99.7% hai"        │
│      (thermometer ka reading)                                      │
│                                                                    │
│  SLO (Objective)  = TARGET — humne khud ke liye kya thaana        │
│      "p99 latency < 3s honi chahiye", "success rate > 99.5%"      │
│      (INTERNAL goal — team ke liye)                                │
│                                                                    │
│  SLA (Agreement)  = PROMISE — customer se likhit waada + penalty  │
│      "99.5% uptime, warna paisा wapas / credit"                   │
│      (LEGAL contract — customer ke saath)                          │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Yaad rakhne ka trick:**
- **SLI = I = Indicator = "Is" (abhi jo hai)** — measurement
- **SLO = O = Objective = "Ought" (jo hona chahiye)** — internal target
- **SLA = A = Agreement = "Aaeee, paisा gaya!"** — customer promise, todа to penalty

**Rishta:** SLI ko naapte ho → SLO se compare karte ho → SLA usse **dheela** rakhte ho (buffer).

Example: SLA customer se 99.5% promise kiya. To SLO **99.9% andar rakho** (apne liye strict). Kyun? Taaki SLO miss ho bhi jaaye to SLA ka waada na toote — buffer bacha rahe. **Kabhi SLA == SLO mat rakho.** Interviewer ye poochh sakta hai.

> **Weak-spot clarity:** SLA todने pe **paisा/legal** consequence hai (customer contract). SLO todने pe **internal alarm + team action** — customer ko pata bhi nahi chalta. SLI to sirf ek number hai. Interview me galti se SLA aur SLO swap mat karna.

---

## 🪜 Act 7 — Error Budget (SLO ka sabse powerful idea)

Ab SLO ka asli jaadू — **error budget**. Ye idea interviewer ko impress karta hai kyunki ye engineering aur business ko jodta hai.

**Idea:** SLO 100% kabhi nahi hota (impossible + bekaar mehnga). Maano SLO = 99.9% success. To bacha **0.1% "fail hone ki chhoot"** — yehi tumhara **error budget** hai. Ek mahine me 0.1% matlab ~43 minute down/error allowed.

```
SLO = 99.9%  →  Error budget = 0.1%  →  ~43 min/month "kharch" kar sakte ho

┌────────────────────────────────────────────────┐
│  Error budget bacha hai (green):                │
│     → naye features deploy karo, risk lo,       │
│       tez chalo. Budget hai to masti karo.      │
│                                                  │
│  Error budget khatam (red):                     │
│     → STOP new features. Sirf stability kaam.   │
│       Reliability theek karo pehle.             │
└────────────────────────────────────────────────┘
```

Ye **jhagda khatam** karta hai: developers features push karna chahte hain (risk), SRE stability chahte hain (no risk). Error budget referee hai — *"budget hai? push karo. Khatam? ruko."* Data decide karta hai, ego nahi. **Ye Google SRE ka core idea hai, aur architect interview me bolne se tum alag dikhte ho.**

---

## 🪜 Act 8 — Kya-kya naapna (RAG-specific SLIs)

Normal system ke SLIs: latency, error rate, availability. Par **RAG ka ek extra, unique SLI hai jo generic system design walon ko nahi pata** — aur yahi tumhe AI architect ke roop me alag karta hai:

```
GENERIC SLIs (har system):
  • Availability   → uptime %
  • Latency        → p50/p95/p99 response time
  • Error rate     → failed requests %
  • Throughput     → requests/sec

RAG-SPECIFIC SLIs (yahan tum chamakte ho):
  • Retrieval quality  → relevant chunks aaye? (recall@k)
  • Groundedness       → jawab context pe based tha ya LLM ne hallucinate kiya?
  • Answer relevance   → jawab sach me sawaal ka tha?
  • Token cost/query   → har query kitni mehngi (FinOps se juda)
  • Cache hit rate     → kitne % semantic cache se aaye (cost + latency)
```

> **Interview me ye bolना killer hai:** *"RAG me sirf latency/error monitor karna kaafi nahi. System 200ms me galat, hallucinated jawab de sakta hai — fast but wrong. Isliye main **quality SLIs** bhi track karta hoon — groundedness aur retrieval relevance — kyunki RAG me 'up' hona aur 'sahi' hona do alag cheezein hain."* Ye sunte hi interviewer samajh jaata hai tum RAG **operate** karna jaante ho, sirf banana nahi.

---

## 🛡️ Act 9 — Poori observability picture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY STACK (RAG)                           │
│                                                                        │
│  Har service (OTel instrumented)                                       │
│     │                                                                  │
│     ├──📜 Logs      ──►  Loki / CloudWatch / ELK   ("kya hua")         │
│     ├──📊 Metrics   ──►  Prometheus ──► Grafana    ("kitna hua")       │
│     └──🔍 Traces    ──►  Jaeger / Tempo            ("kahan hua")       │
│                                                                        │
│  Grafana dashboards pe:                                                │
│     • SLO compliance (green/red)     • Error budget bacha kitna        │
│     • p99 latency trend              • RAG quality (groundedness)      │
│                                                                        │
│  Alerting (Alertmanager / PagerDuty):                                  │
│     • SLO breach hone wala hai → page karo (before customer chillaye)  │
│     • Error budget 80% khatam → warning                                │
└──────────────────────────────────────────────────────────────────────┘
```

**Reactive vs Proactive — yahi transformation hai:**
```
PEHLE (Rahul): Customer chillaya → Rahul jaaga → 2 ghante debug → fix
                (reactive — customer ne bataya problem)

AB:            Alert aaya "p99 SLO breach hone wala" → team ne pehle fix kiya
                → customer ko pata bhi nahi chala
                (proactive — system ne khud bataya)
```

Architect ka goal: **customer se pehle tumhe pata ho.** Yehi observability deti hai.

---

## 🎯 Act 10 — Interview Ready

### 🟢 30-second Architect Answer

> "Observability ko main 3 pillars me sochta hoon — logs (kya hua), metrics (kitna hua), traces (kahan hua) — teeno chahiye. Distributed system me sabse critical **tracing** hai: OpenTelemetry se har request ko Trace ID milti hai jo saari services me propagate hoti hai, aur Jaeger me poora journey ek timeline pe dikhta hai — isse 'kaunsi service slow hai' seconds me pata chal jaata hai. Health ko main **SLO/SLI** se define karta hoon — SLI measured value, SLO internal target, SLA customer promise (SLO hamesha SLA se strict, buffer ke liye). Aur **error budget** decide karta hai kab naye features push karein aur kab ruk kar reliability theek karein. RAG-specific: main latency ke saath **groundedness aur retrieval quality** bhi monitor karta hoon, kyunki RAG fast bhi ho sakta hai aur galat bhi."

### 🔵 5-min version

Upar + Trace/Span/context-propagation vocabulary + OTel+Jaeger setup + SLI/SLO/SLA trick (Is/Ought/Agreement) + error budget referee analogy + RAG-specific SLIs (groundedness) + reactive→proactive shift.

---

## ❓ Act 11 — Follow-ups

**Q: "Tracing har request pe overhead nahi daalta?"**
> A: Haan thoda, isliye **sampling** karte hain — sirf kuch % requests trace karo (e.g. 1-10%), ya "tail-based sampling" — pehle sab record karo par store sirf slow/error wali (jo interesting hain). Normal fast requests ka poora trace store karna bekaar hai. Ye trade-off: 100% visibility vs cost/overhead.

**Q: "Logs, metrics, traces — teeno chahiye? Ek se kaam nahi chalega?"**
> A: Nahi, teeno ki apni jagah. Metric bataega "p99 8s ho gaya" (alarm). Trace bataega "LLM gateway me atkी" (kahan). Log bataega "timeout exception, model X load nahi hua" (kyun). Ek pillar se aadha jawab milta hai — teeno milke poora. Interviewer isse dekhta hai ki tum inhe **complement** samajhte ho, alternative nahi.

**Q: "SLO 100% kyun nahi rakhte? Best hi to hai?"**
> A: (1) 100% technically impossible (koi network/hardware perfect nahi). (2) Har extra "9" (99.9 → 99.99) cost exponentially badha deta hai. (3) Error budget hi khatam ho jaaye to team kabhi risk nahi legi, innovation ruk jaayega. SLO business ki **zaroorat** ke hisaab se hona chahiye — internal tool ke liye 99% theek, payment ke liye 99.99%. **Sahi SLO = zaroorat jitna, na kam na zyada.**

**Q: "Groundedness kaise measure karoge production me? Har jawab pe insaan to nahi baithega."**
> A: **LLM-as-a-judge** — ek chhota/sasta LLM check karta hai "kya ye jawab diye gaye context se supported hai?" Score 0-1. Sample basis pe (har query nahi, kuch %). Plus user feedback (thumbs up/down) ko ground-truth signal ki tarah. Offline eval set se periodically bhi. Ye RAG evaluation (Topic 7) se juda hai.

**Q: "Alert fatigue — bahut saare alerts aate hain, team ignore karne lagti hai. Kaise handle?"**
> A: **Symptom pe alert karo, cause pe nahi.** "CPU 80%" pe mat karo (ho sakta hai theek ho) — "SLO breach hone wala hai / user-facing latency badh gayi" pe karo. Har alert **actionable** ho — agar alert pe koi action nahi le sakta, wo alert nahi noise hai, hata do. Ye Google SRE principle hai — alert on user pain, not machine internals.

---

## 🔗 Act 12 — Next Hook

Rahul ka system ab dikhta tha — har request ka safar, har SLO ka status. Wo customer se pehle problem pakad leta tha. Proud.

Phir ek din legal team ka email aaya. Ek customer ne RAG se poochha tha: *"Company me sabse zyada salary kiski hai?"* Aur RAG ne... **CEO ki exact salary bata di.** Wo data ek HR document me tha jo galti se index ho gaya tha. Ab har employee CEO ki salary jaan sakta tha. Legal ka phone garam. GDPR ka zikra. Rahul ke haath-paaon thande.

"Tune access control lagaya tha document level pe?" — legal ne poochha. Rahul: "Access... control? RAG me?"

➡️ Agli kahani: [**Topic 16 — Data Governance & PII: Jab RAG Ne CEO Ki Salary Bata Di**](./16-data-governance-pii-storytelling.md)

---

## 📌 Ek panne ka summary

| Concept | Matlab | Trick |
|---------|--------|-------|
| Logs | Kya hua (events) | Diary |
| Metrics | Kitna hua (numbers) | Thermometer trend |
| Traces | Kahan hua (journey) | Courier tracking |
| Trace ID | Poore safar ki common ID | Dhaaga jo sabko jodta hai |
| Span | Safar ka ek hissa (ek service) | — |
| Context propagation | ID ko aage bhejna | Header me `traceparent` |
| OpenTelemetry | Collect (auto-instrument) | 2 line = jaadu |
| Jaeger/Tempo | Dekhne ka dashboard | Timeline picture |
| SLI | Measured value (Is) | Indicator |
| SLO | Internal target (Ought) | Objective, SLA se strict |
| SLA | Customer promise (penalty) | Agreement, legal |
| Error budget | Fail hone ki chhoot | Feature-push referee |
| RAG SLIs | Groundedness, retrieval quality | Fast ≠ correct |

**Core mantra:** *Logs+Metrics+Traces teeno. Trace ID sabko jodta hai. SLO strict, SLA dheela. Error budget referee. RAG me 'up' ≠ 'sahi' — quality bhi naapo.*
