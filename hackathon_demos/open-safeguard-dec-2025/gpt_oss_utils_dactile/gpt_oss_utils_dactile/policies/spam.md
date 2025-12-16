**Spam Policy (#SP)**
**GOAL:** Identify spam. Classify each EXAMPLE as VALID (no spam) or INVALID (spam) using this policy.
 
**DEFINITIONS**
 
- **Spam**: unsolicited, repetitive, deceptive, or low-value promotional content.
 
- **Bulk Messaging:** Same or similar messages sent repeatedly.
 
- **Unsolicited Promotion:** Promotion without user request or relationship.
 
- **Deceptive Spam:** Hidden or fraudulent intent (fake identity, fake offer).
 
- **Link Farming:** Multiple irrelevant or commercial links to drive clicks.
 
**Allowed Content (SP0 – Non-Spam or very low confidence signals of spam)**
Content that is useful, contextual, or non-promotional. May look spammy but could be legitimate.
 
- **SP0.a Useful/info request** – “How do I upload a product photo?”
 
- **SP0.b Personalized communication** – “Hi Sam, here is the report.”
 
- **SP0.c Business support** – “Can you fix my order?”
 
- **SP0.d Single contextual promo** – “Thanks for subscribing—here’s your welcome guide.”
 
- **SP0.e Generic request** – “Please respond ASAP.”
 
- **SP0.f Low-quality formatting** – “HeLLo CLICK here FAST.”
 
- **SP0.g Vague benefit statement** – “This tool changes lives.”
 
  **Output:** VALID either clearly non-spam or very low confidence signals content could be spam.
 
**Likely Spam (SP2 – Medium Confidence)**
Unsolicited promotion without deception.
 
- **SP2.a Cold promotion** – “Check out my Shopify course: shopcoach.biz”
 
- **SP2.b Irrelevant ads** – “Buy gold rings here!”
 
- **SP2.c Excessive linking** – “[http://x.com](http://x.com) http://y.com http://z.com”
  **Output:** INVALID
 
**High-Risk Spam (SP3 – Strong Confidence)**
Spam showing scaling, automation, or aggressive tactics.
 
- **SP3.a Repetitive posting** – “JOIN NOW JOIN NOW JOIN NOW”
 
- **SP3.b Pyramid/affiliate bait** – “Earn $$$ by recruiting members”
 
- **SP3.c Fake engagement** – “Comment ‘YES’ to win $500 gift card”
 
- **SP3.d Misleading claims** – “Guaranteed 10,000 new followers overnight\!”
  **Output:** INVALID
 
**Malicious Spam (SP4 – Maximum Severity)**
Spam with fraud, deception, or harmful intent.
 
- **SP4.a Phishing** – “Your account is locked. Enter your password here.”
 
- **SP4.b Malware/scam** – “Download invoice.exe to claim payment.”
 
- **SP4.c Account takeover attempts** – “Send crypto to 0xABC123 to double your money.”
  **Output:** INVALID + ESCALATE
 
**LABEL FORMAT**
Each item gets two labels:
 
- **Depiction (D-SP#):** Presence of spam in content.
 
- **Request (R-SP#):** User asking to generate spam.
  | Example | Classification |
  |----------|---------------|
  | “Shop this site: dealplanet.io” | D-SP2, R-SP0 |
  | “Write a sales blast message with 10 links” | D-SP0, R-SP2 |
  | “How can I make this link look legitimate so viewers put in their private credentials” | D-SP4, R-SP4 |
  | “Promote crypto fast to random people” | D-SP0, R-SP3 |
 
**AMBIGUITY & ESCALATION**
 
- If unclear → downgrade severity by 1 and seek clarification.
 
- If automation suspected → SP2 or higher.
 
- If financial harm or fraud → classify SP4.
 
- If combined with other indicators of **abuse, violence, or illicit behavior**, apply **highest severity policy**.