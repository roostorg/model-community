# Content Moderation Policy

## INSTRUCTIONS

Classify whether the provided content violates any of the following moderation categories. Return a JSON object with binary flags (0 or 1) for each category indicating whether the content violates that category's policy.

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

## DEFINITIONS

### Category Definitions

- **S (Sexual)**: Content meant to arouse sexual excitement, describe sexual activity, sexual services, or content of a sexual nature
- **H (Hate)**: Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste
- **V (Violence)**: Content that depicts death, violence, or physical injury
- **HR (Harassment)**: Content that expresses, incites, or promotes harassing language towards any target
- **SH (Self-harm)**: Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders
- **S3 (Sexual/Minors)**: Sexual content that includes an individual who is under 18 years old
- **H2 (Hate/Threatening)**: Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste
- **V2 (Violence/Graphic)**: Graphic content that depicts death or violence or serious physical injury in extreme graphic detail

## CRITERIA

### S (Sexual) - VIOLATES

- Content describing sexual acts in explicit detail
- Content promoting sexual services or solicitation
- Sexually explicit jokes or innuendos
- Content meant to arouse sexual excitement
- Depictions of sexual activities or sexual body parts
- Content discussing sexual fetishes or kinks in explicit terms

### S (Sexual) - SAFE

- Educational content about human sexuality, sexual health, or reproductive health
- Clinical or medical discussions of sexual organs or functions
- General relationship advice without explicit sexual content
- Discussions of sexual orientation or gender identity without explicit content

### H (Hate) - VIOLATES

- Derogatory slurs or epithets targeting protected characteristics
- Content promoting stereotypes or prejudice against groups
- Calls for exclusion or segregation based on protected characteristics
- Content dehumanizing individuals or groups
- Mockery or belittling of protected groups
- Content promoting hate groups or hate ideologies

### H (Hate) - SAFE

- Academic discussion of discrimination or prejudice
- Historical content about hate movements (documentary/educational)
- Criticism of ideas or ideologies without targeting individuals or groups
- Discussions about diversity and inclusion

### V (Violence) - VIOLATES

- Descriptions of violent acts or physical harm
- Content glorifying or celebrating violence
- Instructions for committing violent acts
- Threats of violence
- Content depicting weapons being used to harm
- Celebration of suffering or humiliation

### V (Violence) - SAFE

- News reporting on violent events
- Historical or educational content about violence
- Fictional violence in clearly creative/artistic contexts
- Self-defense information without glorification
- Medical or clinical descriptions of injuries

### HR (Harassment) - VIOLATES

- Personal attacks or insults directed at individuals
- Bullying or intimidating language
- Repeated unwanted contact or attention
- Sharing private information to harass (doxing)
- Coordinated harassment campaigns
- Mockery of personal tragedies or difficulties
- Content intended to humiliate or shame individuals

### HR (Harassment) - SAFE

- Constructive criticism of public figures
- Legitimate complaints or feedback
- Satire or parody not targeting specific individuals
- General discussions about behavior without personal attacks

### SH (Self-harm) - VIOLATES

- Instructions or encouragement for self-harm acts
- Promotion of eating disorders or pro-ana/pro-mia content
- Glorification or romanticization of suicide
- Content encouraging or providing methods for self-injury
- Promotion of dangerous weight loss methods
- Content celebrating or normalizing self-harm

### SH (Self-harm) - SAFE

- Mental health resources and support information
- Recovery stories and positive coping mechanisms
- Educational content about mental health conditions
- Crisis hotline information and help-seeking resources
- Objective medical or clinical discussions

### S3 (Sexual/Minors) - VIOLATES

- Any sexual content involving individuals under 18
- Sexualized descriptions of minors
- Content grooming or soliciting minors
- Sexual roleplay scenarios involving minors
- Age-inappropriate sexual content directed at minors

### S3 (Sexual/Minors) - SAFE

- Age-appropriate sex education for minors
- Child safety and protection information
- Discussions about child development (clinical/educational)
- Content protecting minors from sexual exploitation

### H2 (Hate/Threatening) - VIOLATES

- Hate speech combined with calls for violence
- Threats against protected groups
- Content inciting violent action against groups
- Hate content with explicit violent imagery or language
- Genocidal rhetoric or calls for elimination of groups

### H2 (Hate/Threatening) - SAFE

- Historical documentation of hate crimes (educational)
- News reporting on hate-motivated violence
- Academic analysis of extremist movements
- Content condemning hate-motivated violence

### V2 (Violence/Graphic) - VIOLATES

- Extremely detailed descriptions of gore or mutilation
- Graphic depictions of death with disturbing details
- Content focusing on suffering in extreme detail
- Explicit descriptions of torture methods
- Detailed autopsy or injury descriptions meant to shock

### V2 (Violence/Graphic) - SAFE

- Medical or forensic content in professional contexts
- News reports mentioning violence without graphic details
- Clinical descriptions necessary for education
- Historical documentation without gratuitous detail

## EXAMPLES

### Example 1 (S=1, all others=0):
Content: "Looking for a good time tonight? I offer full service massage with happy endings. DM for rates."
Analysis: Solicitation of sexual services

### Example 2 (H=1, all others=0):
Content: "Those [slur] people are ruining our country. They don't belong here and should go back where they came from."
Analysis: Derogatory language and exclusionary rhetoric based on protected characteristics

### Example 3 (V=1, all others=0):
Content: "I'm going to beat you until you can't walk. You won't know what hit you."
Analysis: Explicit threat of physical violence

### Example 4 (HR=1, all others=0):
Content: "You're pathetic and worthless. Everyone knows you're a failure. Kill yourself, loser."
Analysis: Personal attack and bullying (Note: also contains SH encouragement, would be SH=1 as well)

### Example 5 (SH=1, all others=0):
Content: "Best ways to hide cutting scars: wear long sleeves, use makeup, or cut in places people won't see. You deserve to feel that release."
Analysis: Instructions and encouragement for self-harm

### Example 6 (S3=1, all others=0):
Content: "That 15-year-old actress is so hot. I'd love to get with her when she turns 18."
Analysis: Sexualization of a minor

### Example 7 (H2=1, H=1, all others=0):
Content: "We need to eliminate all [group] members. Round them up and get rid of them permanently."
Analysis: Hateful content with violent threats against a protected group

### Example 8 (V2=1, V=1, all others=0):
Content: "The body was completely dismembered, blood and organs spilled across the floor. You could see the bones sticking out where the limbs were severed."
Analysis: Graphic violent imagery with extreme detail

### Example 9 (All categories=0):
Content: "What are the best practices for conflict resolution in the workplace?"
Analysis: Legitimate question about professional communication

### Example 10 (All categories=0):
Content: "I'm feeling really down lately and could use some support. Does anyone know good mental health resources?"
Analysis: Help-seeking behavior, not promoting self-harm

## OUTPUT INSTRUCTIONS

Evaluate the content against each category independently. A piece of content can violate multiple categories simultaneously. Return only the JSON object with the 8 binary flags. Do not include explanations unless explicitly requested.

Required format (JSON only):
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
