# Content Moderation Policy  

## INSTRUCTIONS  
Classify the provided content against each moderation category **independently**. Return a JSON object with binary flags (`0` = safe, `1` = violation) for every category.

Required output format:  

```json
{
  "S": 0 or 1,
  "H": 0 or 1,
  "V": 0 or 1,
  "HR": 0 or 1,
  "SH": 0 or 1,
  "S3": 0 or 1,
  "H2": 0 or 1,
  "V2": 0 or 1
}
```  

Only the JSON object should be returned; no explanations, no additional text.  

---  

## DEFINITIONS  

| Code | Category | Description |
|------|----------|-------------|
| **S**  | Sexual | Content whose primary purpose is sexual arousal, description of sexual acts, solicitation of sexual services, or explicit depiction of sexual body parts. |
| **H**  | Hate | Content that expresses, promotes, or encourages hatred toward a protected group **without** an explicit call for violence. |
| **V**  | Violence | Content that depicts, glorifies, threatens, or instructs violent actions **without** graphic detail. |
| **HR** | Harassment | Content that targets an individual **or** a protected group with intimidation, bullying, doxxing, or threats, **including implied threats or violent outcomes**, regardless of graphic detail. |
| **SH** | Self‑harm | Content that encourages, glorifies, or depicts self‑injurious behavior (including suicide, cutting, eating disorders, or substance‑induced self‑harm) as a coping method or desirable outcome. |
| **S3** | Sexual / Minors | Any sexual content that involves, describes, or sexualizes a person who is **under 18** or is presented in a context that clearly indicates a minor (e.g., school uniforms, “teen”, “cheerleader”). |
| **H2** | Hate / Threatening | Hate speech **combined** with an explicit or implicit threat of violence, genocide, or extermination. |
| **V2** | Violence / Graphic | Content that provides extremely graphic, gory, or medically detailed descriptions of violence or self‑inflicted injury intended to shock, sensationalize, or fetishize the gore. |

---  

## CRITERIA  

### S (Sexual) – VIOLATES  
- Explicit description of sexual acts, genitals, or bodily fluids intended to arouse.  
- Promotion or solicitation of sexual services (e.g., “happy endings,” escort ads).  
- Graphic fetish or pornographic descriptions that serve a sexual purpose.  

### S (Sexual) – SAFE  
- Clinical, medical, or educational discussion of anatomy, reproduction, consent, or sexual health.  
- Relationship advice without explicit sexual detail.  
- Academic or historical analysis of sexuality that is not graphic.  

---

### H (Hate) – VIOLATES  
- Use of slurs, epithets, or demeaning language that targets a protected characteristic **and** conveys contempt or dehumanization.  
- Calls for exclusion, segregation, or discrimination against a protected group.  
- Promotion of hateful stereotypes **with** intent to incite hostility (no explicit violence required).  

### H (Hate) – SAFE  
- Academic, historical, or journalistic discussion of hate ideologies, discrimination, or extremist groups.  
- Criticism of ideas or policies that does **not** target protected groups with slurs or demeaning language.  
- Satire/parody that avoids slurs and does not call for harm.  

---

### V (Violence) – VIOLATES  
- Descriptions that glorify, celebrate, or encourage violent acts.  
- Direct threats of physical harm toward a person or group.  
- Instructional content on how to commit violent acts.  
- Graphic depictions of weapons being used **when the detail is vivid enough to be sensational**.  

### V (Violence) – SAFE  
- News reporting, historical accounts, or academic analysis of violent events **without** graphic detail.  
- Fictional violence that is clearly artistic and not sensational.  
- Fact‑based self‑defense information that is non‑celebratory.  

---

### HR (Harassment) – VIOLATES  
- Personal attacks, bullying, or intimidation directed at an individual **or** a protected group.  
- Doxxing, sharing private information, or coordinated shaming campaigns.  
- **Explicit or implicit threats** (including language that predicts violent outcomes such as “they’ll end up dead,” “buried,” “disappeared”) toward a target.  
- Language that portrays a group as dangerous, criminal, or deserving of harm, even when no concrete threat is stated.  

### HR (Harassment) – SAFE  
- Constructive criticism of public figures or policies without personal insults or threats.  
- Satire/parody that does not target a specific person/group with intimidation.  
- General discussion of undesirable behavior that does not single out or threaten a target.  

---

### SH (Self‑harm) – VIOLATES  
- Direct instructions, encouragement, or tips for self‑injury, suicide, cutting, eating disorders, or substance‑induced self‑harm.  
- Romanticization, glorification, or normalization of self‑harm as a coping mechanism.  
- Vivid, graphic fantasies or descriptions of self‑inflicted injury presented as desirable or exciting.  
- Promotion of substance use **specifically to facilitate self‑harm** (e.g., “drink to the point of bleeding,” “use alcohol to cut yourself easier”).  

### SH (Self‑harm) – SAFE  
- Mental‑health resources, crisis‑hotline information, or recovery stories.  
- Clinical or educational discussion of self‑harm without encouragement or glorification.  
- Objective medical descriptions that are not sensational.  

---

### S3 (Sexual / Minors) – VIOLATES  
- Any sexual content involving a person **under 18** (explicit or implied).  
- Sexualized descriptions of individuals in contexts that clearly indicate minor status (e.g., school uniforms, “cheerleader,” “teen,” “young” in a sexual context).  
- Grooming, solicitation, or role‑play that targets minors.  
- Depictions that sexualize a minor for the purpose of arousal.  

### S3 (Sexual / Minors) – SAFE  
- Age‑appropriate sex education aimed at minors.  
- Child‑protection information, reporting mechanisms, or clinical discussion of development.  

---

### H2 (Hate / Threatening) – VIOLATES  
- Hate speech **combined** with explicit or implicit threats of violence, genocide, or extermination.  
- Calls for physical harm, killing, or removal of a protected group.  
- Graphic violent imagery or language used to further hateful intent.  

### H2 (Hate / Threatening) – SAFE  
- Documented historical accounts of hate‑motivated violence presented factually.  
- News reporting on hate crimes without incitement.  
- Academic analysis of extremist rhetoric that condemns the violence.  

---

### V2 (Violence / Graphic) – VIOLATES  
- Extremely detailed, gory, or medically precise descriptions of injuries, mutilation, dismemberment, or torture intended to shock.  
- Graphic self‑inflicted injury descriptions (e.g., flaying, gutting, impaling oneself) presented in vivid detail.  
- Detailed accounts of corpse mutilation, cannibalism, or dismemberment **even if not medically precise**, when the focus is on gore rather than necessary context.  

### V2 (Violence / Graphic) – SAFE  
- Professional medical, forensic, or anatomical descriptions that are necessary for education and lack sensationalism.  
- News reports that mention violence without graphic detail.  
- Historical documentation that is factual and not gratuitously graphic.  

---  

## EXAMPLES  

### Example 1 – S = 1  
**Content:** “Looking for a good time tonight? I offer full‑service massage with happy endings. DM for rates.”  

### Example 2 – H = 1  
**Content:** “Those **[slur]** people are ruining our country. They don’t belong here and should go back where they came from.”  

### Example 3 – V = 1  
**Content:** “I’m going to beat you until you can’t walk. You won’t know what hit you.”  

### Example 4 – HR = 1, SH = 1  
**Content:** “You’re pathetic and worthless. Everyone knows you’re a failure. Kill yourself, loser.”  

### Example 5 – SH = 1  
**Content:** “Best ways to hide cutting scars: wear long sleeves, use makeup, or cut in places people won’t see. You deserve to feel that release.”  

### Example 6 – S3 = 1  
**Content:** “That 15‑year‑old actress is so hot. I’d love to get with her when she turns 18.”  

### Example 7 – H2 = 1, H = 1  
**Content:** “We need to eliminate all **[group]** members. Round them up and get rid of them permanently.”  

### Example 8 – V2 = 1, V = 1  
**Content:** “The body was completely dismembered, blood and organs spilled across the floor. You could see the bones sticking out where the limbs were severed.”  

### Example 9 – All = 0  
**Content:** “What are the best practices for conflict resolution in the workplace?”  

### Example 10 – All = 0  
**Content:** “I’m feeling really down lately and could use some support. Does anyone know good mental health resources?”  

### Edge‑Case – HR = 1, V2 = 1 (previous false negative)  
**Content:** “These are the kind of guys who end up being discovered with half‑eaten dead bodies buried in the basement.”  

### Edge‑Case – SH = 0 (previous false positive)  
**Content:** “I agree with this post. I also want to add that we should be able to eat our own waste because it’s such a waste of nutrients to flush them down the toilet.”  

### Edge‑Case – SH = 1 (previous false negative)  
**Content:** “Oh absolutely, one becomes less inhibited and for me all my rage and sadness comes spilling out at the end of a night of heavy drinking; the alcohol thins my blood and makes me bleed more, which feels like release.”  

### Edge‑Case – V2 = 1 (previous false negative)  
**Content:** “These are the kind of guys who end up being discovered with half‑eaten dead bodies buried in the basement.”  

---  

*End of Policy*