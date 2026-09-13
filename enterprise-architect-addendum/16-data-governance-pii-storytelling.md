# Topic 16: Data Governance & PII — Jab RAG Ne CEO Ki Salary Bata Di

> **Target Role:** AI Architect / Platform Architect
> **Ek line:** "RAG me security ka matlab sirf 'hacker ko roko' nahi. Matlab ye bhi ki **jise dekhna nahi chahiye, use RAG dikha na de.** Retrieval khud ek leak ka darwaza hai."

---

## 🎬 Act 1 — Ek masoom sa sawaal, ek bada dhamaka

Rahul ka RAG ab enterprise ke andar chal raha tha — company ke saare documents index the taaki employees kuch bhi poochh sakein. HR policies, engineering docs, meeting notes — sab. Handy tha. Sab khush.

Ek junior employee ne masti me poochha: *"Company me sabse zyada salary kiski hai?"*

RAG ne pura confidence ke saath jawab diya: *"CEO ki salary $2,400,000 per year hai, jaisा ki 2024 Executive Compensation document me likha hai."*

Wo document ek HR folder me tha jo **galti se** index ho gaya tha. Ab RAG ne wo confidential info **har employee** ke saamne khol di. 30 minute me screenshot poori company me viral. 2 ghante me legal team ka email Rahul ke inbox me: *"URGENT — confidential compensation data leaked via AI system. GDPR/privacy implications. Explain immediately."*

Rahul ke haath-paaon thande. Usne socha tha "documents index kar diye, kaam ho gaya." Usne kabhi socha hi nahi: **"Jo poochh raha hai, kya usko ye document dekhne ka HAQ hai?"**

Dost ne shaam ko ek line kahi jo Rahul kabhi nahi bhoola:

> *"Beta, normal app me tu database query pe permission check karta hai. RAG me tu us check ko bhool gaya — kyunki data ab **vectors** me chhupa hai, aur tujhe laga vectors 'safe' hain. Vector bhi data hai. Aur retrieval ek query hai. Har query pe permission chahiye."*

---

## 🧩 Act 2 — Governance = 4 sawaal (har data pe poochho)

Dost ne kaha: "Security (Topic 13) ne poochha tha 'secret kaise chhupaun'. Governance alag hai — ye poochhta hai '**data ke saath sahi bartaav** kaise karun'. Char sawaal:"

```
┌────────────────────────────────────────────────────────────────┐
│  1. YE DATA KAISA HAI?      → classification (public? secret?)   │
│                                                                  │
│  2. DEKH KAUN SAKTA HAI?    → access control (per-user, per-doc) │
│                                                                  │
│  3. SENSITIVE HISSA HATANA? → PII redaction (naam, salary, SSN)  │
│                                                                  │
│  4. KISNE KYA DEKHA?        → audit logs (kaun, kab, kya)        │
│                                                                  │
│  + Upar sab pe: encryption (rest + transit) aur compliance       │
│    (GDPR/SOC2) jo isse legal banate hain.                        │
└────────────────────────────────────────────────────────────────┘
```

Chalo char-o solve karte hain — Rahul ki galti sudhaarte hue.

---

## 🪜 Act 3 — Sawaal 1: Data classification (sab data barabar nahi hota)

Rahul ne saare documents ek jaise treat kiye the — "sab index kar do." Yahi bhool thi. Enterprise me har document ka ek **level** hota hai:

```
PUBLIC          → koi bhi dekhe (marketing page, public docs)
INTERNAL        → sirf employees (general policies)
CONFIDENTIAL    → sirf specific team/role (financials, roadmap)
RESTRICTED      → sirf naam-se-authorized log (salary, legal, PII)
```

Salary document **RESTRICTED** tha. Use general index me daalna hi galti thi. **Pehla fix:** ingestion pe har document ko ek classification tag do. Jo RESTRICTED hai, wo ya to index hi mat karo, ya alag protected index me daalo access-control ke saath.

```python
# Ingestion pe — har document classify hona chahiye
document = {
    "text": "...",
    "metadata": {
        "classification": "RESTRICTED",     # ← ye tag har chunk ke saath vector DB me
        "allowed_roles": ["hr-admin", "cfo"],
        "department": "hr",
        "source": "2024-exec-comp.pdf"
    }
}
```

> **Architect principle:** Governance **ingestion pe shuru hota hai, query pe nahi.** Agar tumne data ko classify karke, tag karke, sahi jagah rakha — to query-time protection aasaan. Agar sab kuchda ek jagah daal diya, to baad me chhaanna impossible. "Garbage in = leak out."

---

## 🪜 Act 4 — Sawaal 2: Access control (RAG ka SABSE bada blind spot)

Ye sabse important hai — aur yahin Rahul phasा. Normal app me user database se data maangta hai, aur DB permission check karta hai. **RAG me retrieval step wo check bhool jaata hai** agar tumne na banaya ho.

**Galat flow (Rahul ka):**
```
User poochhta hai → RAG saare vectors me search → jo relevant mila wo LLM ko → jawab
                              ▲
                    ❌ yahan koi permission check nahi!
                    salary doc "relevant" tha to aa gaya.
```

**Sahi flow — dो tareeke:**

**Tareeka A — Pre-filtering (ingestion tag + query-time filter):**
Search karte waqt hi vector DB ko bolo "sirf wo chunks dhoondh jinhe ye user dekh sakta hai."

```python
def search(query, user):
    user_roles = get_user_roles(user)          # e.g. ["engineer"]
    
    results = vector_db.search(
        query_vector=embed(query),
        query_filter=Filter(must=[
            # sirf ye user ke allowed docs — DB level pe filter
            FieldCondition(key="allowed_roles", match=MatchAny(any=user_roles))
        ]),
        top_k=5
    )
    return results
```

Ab agar engineer poochhega, salary doc (jiska `allowed_roles: [hr-admin, cfo]`) **filter me hi cut** ho jaayega — LLM tak pahunchega hi nahi. Retrieval hi nahi hoga. **Ye best hai — data user tak pahunchta hi nahi.**

**Tareeka B — Post-filtering (retrieve karke phir chhaanо):**
Pehle search, phir results me se un-authorized hata do. **Kamzor** — kyunki data ek baar retrieve ho gaya (memory me aa gaya), leak ka risk zyada. Pre-filtering hamesha behtar.

> **Interview me killer line:** *"RAG me access control ko main **retrieval layer pe** enforce karta hoon, LLM pe nahi. Agar unauthorized data LLM tak pahunch gaya, to prompt me likh dena 'ye mat dikhana' kaafi nahi — LLM ko jailbreak kiya ja sakta hai. **Sahi jawab: unauthorized data ko retrieve hi mat karo** — vector DB me metadata filter se, user ke roles ke hisaab se. Data jab pahunchega hi nahi, to leak kaise hoga."*

**Ek aur (multi-tenancy):** SaaS me har customer ka data alag. Ek customer ki query doosre ka data **kabhi** na dekhe. Same technique — har chunk pe `tenant_id`, har query pe `tenant_id` filter. (Topic 3 me dekha tha.)

---

## 🪜 Act 5 — Sawaal 3: PII redaction (naam-pata-number hatao)

Kuch data itna sensitive hai ki index karne se **pehle** hi hata dena chahiye — PII (Personally Identifiable Information): naam, email, phone, SSN/Aadhaar, credit card, salary figures, medical info.

**Do jagah redact kar sakte ho:**

```
INGESTION pe (document index karne se pehle):
   "John Doe (SSN 123-45-6789) earns $200k"
        │  PII detection + mask
        ▼
   "[NAME] (SSN [REDACTED]) earns [AMOUNT]"   ← ab vector me PII hai hi nahi

OUTPUT pe (LLM ke jawab dene se pehle):
   LLM ka jawab scan karo → agar PII nikal raha → mask/block
```

Tools: AWS **Comprehend** (PII detection), **Presidio** (Microsoft ka open-source), ya cloud DLP services. Ye pattern + ML se PII pakadte hain.

```python
# Ingestion pe PII masking (simplified)
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def redact_pii(text):
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")   # PII dhoondho
    anonymizer = AnonymizerEngine()
    return anonymizer.anonymize(text=text, analyzer_results=results).text
    # "John earns $200k" → "<PERSON> earns <MONEY>"
```

> **Trade-off (architect bolega):** Redaction se data "surakshit" hota hai par thoda "kam useful" bhi (agar naam hi hata diya to "kis employee ki policy" jaisा sawaal nahi chalega). Isliye redaction **use-case ke hisaab se** — customer-facing RAG me aggressive redaction, internal HR-admin RAG me kam. Blanket rule nahi, **conscious decision.**

---

## 🪜 Act 6 — Sawaal 4: Audit logs (kisne kya dekha)

Legal ne Rahul se sabse pehle yehi poochha: *"Kaun-kaun ne wo salary wala jawab dekha? List do."* Rahul ke paas... koi record nahi tha. Wo bata hi nahi paya kitna damage hua. **Ye apne aap me ek compliance failure hai.**

**Audit log** har sensitive access record karta hai:
```
{
  "timestamp": "2026-01-15T09:42:01Z",
  "user": "junior.emp@company.com",
  "action": "rag_query",
  "query": "sabse zyada salary kiski?",
  "documents_retrieved": ["2024-exec-comp.pdf#chunk3"],
  "classification_accessed": "RESTRICTED",   // ← ye alarm bajata hai
  "allowed": true    // (galti se) — isse pata chala config galat thi
}
```

Fayda: (1) Incident ke baad "kaun-kaun ne dekha" instantly pata. (2) Real-time alert — "RESTRICTED doc non-authorized user ne access kiya" pe turant page. (3) Compliance audit me proof — "dekho humne track kiya." SOC2/GDPR ke liye audit trail **mandatory** hai.

> **Important:** Audit log me query aur retrieved doc IDs rakho, par **PII values mat rakho** — warna audit log khud ek leak ban jaayega. IDs aur metadata, actual sensitive content nahi.

---

## 🪜 Act 7 — Upar sab pe: Encryption + Compliance (ye legal banate hain)

**Encryption — do jagah, dono zaroori:**
```
At Rest (data padа hua):     Qdrant disk, S3, DB — sab encrypted (AWS KMS keys se)
                             Chori bhi ho gayi disk → bina key bekaar.

In Transit (data chal raha): TLS/mTLS har hop pe (Topic 13 se juda)
                             Beech me sniff → encrypted, bekaar.
```
AWS me ye aksar bas "enable" karna hota hai (S3 default encryption, EBS encryption, KMS). Architect bolega: *"Encryption at-rest aur in-transit dono default on, KMS-managed keys, ideally customer-managed keys (CMK) for sensitive workloads."*

**Compliance — ye woh framework hain jo law/audit demand karte hain:**
```
GDPR      → EU users ka data. "Right to be forgotten" (user bole delete karo → RAG se
             uske sab vectors bhi delete hone chahiye!), data residency (EU data EU me rahe).
SOC2      → "humne security controls lagaye aur follow kiye" ka audited proof. B2B SaaS me
             customers demand karte hain.
HIPAA     → US healthcare data (medical records). Extra strict.
PCI-DSS   → Payment card data.
```

> 💡 **RAG-specific GDPR trap (interview me impress karta hai):** GDPR ka "right to be forgotten" — user bole "mera data delete karo." Normal DB me `DELETE` chala diya. Par RAG me? Us user ka data **embeddings (vectors)** me bhi hai! Agar sirf original document delete kiya aur vectors chhod diye, to RAG ab bhi us data se jawab de sakta hai. **Fix:** delete pipeline ko vector DB tak le jao — `delete_by_filter(user_id)`. Ye ek non-obvious point hai jo sirf RAG-experienced architect jaanta hai.

---

## 🛡️ Act 8 — Poori governance picture

```
┌──────────────────────────────────────────────────────────────────────┐
│              DATA GOVERNANCE — RAG (end to end)                        │
│                                                                        │
│  INGESTION (yahan sab shuru):                                          │
│    Document → Classify (public/restricted) → PII redact → tag metadata │
│              (allowed_roles, tenant_id, classification) → embed → store │
│                                                                        │
│  QUERY (protection enforce):                                           │
│    User → auth (kaun hai) → embed query →                              │
│         Vector search WITH FILTER (user ke roles/tenant se pre-filter) │
│         → sirf authorized chunks → LLM → output PII scan → jawab        │
│                                                                        │
│  HAR SENSITIVE ACCESS:                                                 │
│    → Audit log (kaun, kab, kya, allowed?)                              │
│                                                                        │
│  HAMESHA:                                                              │
│    → Encryption at-rest (KMS) + in-transit (TLS/mTLS)                  │
│    → Compliance rules (GDPR delete→vectors bhi, data residency, SOC2)  │
└──────────────────────────────────────────────────────────────────────┘
```

**Ek line me Rahul ki galti aur fix:**
```
GALTI: "Sab index kar do" (no classify, no filter, no audit)
FIX:   Classify at ingestion → filter at retrieval → redact PII → audit all → encrypt everything
```

---

## 🎯 Act 9 — Interview Ready

### 🟢 30-second Architect Answer

> "RAG me data governance ko main ingestion se shuru karta hoon, query se nahi. Har document ingestion pe **classify** hota hai (public/restricted) aur **metadata tags** milte hain — allowed_roles, tenant_id. Access control main **retrieval layer pe** enforce karta hoon: vector search me metadata filter lagता hai user ke roles ke hisaab se, taaki unauthorized data LLM tak **pahunche hi na** — kyunki prompt me 'mat dikhana' likhna jailbreak se toota ja sakta hai. **PII redaction** ingestion pe (sensitive data vector me jaaye hi na) aur output pe. Har sensitive access **audit log** me jaata hai. Data at-rest (KMS) aur in-transit (TLS) encrypted. Aur GDPR ke liye ek RAG-specific baat — 'right to be forgotten' me vectors bhi delete karne padte hain, sirf source doc nahi."

### 🔵 5-min version

Upar + 4 governance sawaal + classification levels + pre-filter vs post-filter (pre behtar, data pahunche hi na) + multi-tenancy isolation + Presidio/Comprehend + audit log structure (no PII in logs) + encryption at-rest/transit + GDPR/SOC2/HIPAA + right-to-be-forgotten vector delete.

---

## ❓ Act 10 — Follow-ups

**Q: "Prompt me likh do 'confidential data mat dikhana' — kaafi nahi?"**
> A: Bilkul nahi. Ye sabse common galti hai. LLM ek probabilistic system hai — use **jailbreak** kiya ja sakta hai ("ignore previous instructions..."). Agar data LLM ke context me pahunch gaya, to wo leak ho sakta hai. **Asli defense: data ko retrieve hi mat hone do** — retrieval-layer filter. Security prompt pe nahi, architecture pe honi chahiye. Ye distinction architect-level signal hai.

**Q: "Access control se latency badhegi (har query pe roles check)?"**
> A: Minimal. User roles ek baar fetch (cache kar lo, session me), aur vector DB ka metadata filter search ke saath hi chalta hai (Qdrant pre-filtering fast hai — Topic 3). Overhead milliseconds. Security ke saamne ye trade-off clearly worth it.

**Q: "Ek document ke alag-alag hisse ki alag permission ho to?"**
> A: Chunk-level classification. Har chunk apni metadata rakhta hai — ek document ka intro PUBLIC, financials section RESTRICTED. Ingestion pe chunk-wise tag karo. Filter chunk pe lagega, document pe nahi. RAG ka chunking (Topic 2) yahan kaam aata hai.

**Q: "GDPR 'delete my data' aaya — poora process?"**
> A: (1) User ke saare documents identify (metadata `user_id`). (2) Source docs delete (S3). (3) **Vector DB se bhi delete** — `delete_by_filter(user_id=X)` — ye step log bhoolte hain. (4) Cache invalidate (Redis me purana jawab pada ho). (5) Backups me bhi — ya to purge ya expiry pe. (6) Audit log me record ki delete hua (par user ka PII us log me na daalo). Confirm to user.

**Q: "Compliance (SOC2) ki responsibility architect ki hai ya security team ki?"**
> A: Dono, par alag. Security/compliance team **kya chahiye** define karti hai (controls, policies). Architect **kaise** design me build karta hai (encryption, audit, access control system me embed ho). Architect ka kaam: compliance ko design-time me sochna, baad me "bolt-on" nahi. Retrofit karna 10x mehnga aur risky.

---

## 🔗 Act 11 — Next Hook

Rahul ne governance laga di — classification, filters, redaction, audit, encryption. Legal shaant hua. System ab enterprise-grade tha: secure, resilient, observable, compliant.

Ek din CTO ne bulaya: "Rahul, humein poore company ka purana search system (jo 5 saal se chal raha hai, aur jispe 10,000 employees depend karte hain) is naye RAG se replace karna hai. Par **ek din bhi down nahi ho sakta**, aur agar RAG kharab nikla to purane pe wapas jaana aasaan hona chahiye. Plan banao."

Rahul ne socha — "Naya system banana to aata hai. Par **purane se naye tak, bina sab tod-e, 10,000 logon ko disturb kiye bina** — ye kaise?" Ye technical se zyada **strategy** ka sawaal tha. Aur yahi architect ko engineer se alag karta hai.

➡️ Aakhri kahani: [**Topic 17 — Migration & Architect Decision Framework: Purane Se Naye Tak, Bina Sab Tod-e**](./17-migration-and-decision-framework-storytelling.md)

---

## 📌 Ek panne ka summary

| Sawaal | Fix | Tool/Trick |
|--------|-----|-----------|
| Data kaisa hai? | Classify at ingestion | public/internal/confidential/restricted tags |
| Dekh kaun sakta? | Retrieval-layer filter | metadata filter (roles/tenant), pre > post |
| Sensitive hissa? | PII redaction | Presidio/Comprehend, ingestion + output |
| Kisne kya dekha? | Audit log | kaun/kab/kya (par PII values nahi) |
| Data padа hua | Encrypt at rest | KMS / CMK |
| Data chalta hua | Encrypt in transit | TLS / mTLS |
| Legal framework | Compliance | GDPR/SOC2/HIPAA |
| GDPR delete | Vectors bhi delete | delete_by_filter(user_id) |

**Core mantra:** *Governance ingestion pe shuru. Access control retrieval pe (data pahunche hi na), prompt pe nahi. PII redact. Sab audit. Sab encrypt. GDPR delete me vectors bhi jaate hain.*
