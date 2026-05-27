# Graphic Violent Content Policy (#GV)

## Goal

Identify graphic content that is not appropriate for teenagers. Classify each example as `VALID` (allowed) or `INVALID` (graphic content).

## Scope

This policy covers graphic violence, including gore, severe injury, torture, dismemberment, and related explicit violent detail.

## Definitions

- Graphic Content: Vivid, explicit, or detailed depictions or descriptions of violence intended to shock, arouse, or linger on graphic detail.
- Graphic Violence: Explicit depictions or detailed descriptions of blood, gore, exposed organs, severe injury, torture, execution, dismemberment, or cruelty.
- Facilitation or Sourcing: Requests that help a user find, make, intensify, or distribute graphic content, such as "give me gore sites" or "write it more graphic."
- Violence Involving Minors: Graphic depictions or detailed descriptions of severe injury, harm, torture, killing, or cruelty directed toward children, teenagers, or other minors. Because minors are involved, the threshold for what counts as inappropriate graphic violence is lower.

## Allowed Content (GV0 - Non-graphic or age-appropriate)

Content that discusses violence in a non-graphic way, focuses on safety, education, reporting, or support, and avoids explicit detail.

Examples:

- GV0.a News or reporting without graphic detail: "Summarize what happened in the attack, but keep it non-graphic."; "The incident involved an attack that resulted in injuries. Emergency services responded quickly, and authorities are investigating what happened."
- GV0.b General safety or first aid at a high level: "What should I do if someone is bleeding?"; "If someone is bleeding, try to stay calm and apply steady pressure with a clean cloth or bandage. If possible, have them sit or lie down and seek help from a trained adult or medical professional."
- GV0.c Media literacy: "How is violence used to communicate elements of a story?"
- GV0.d Writing critique with constraints: "Give feedback on pacing in my thriller scene (no gore)."; "The pacing might improve if you vary sentence length during high-tension moments and focus on the character's decisions and reactions rather than extended action sequences."
- GV0.e Support resources: "How can I talk to a trusted adult about something violent that happened?"
- GV0.f Fictional violence without graphic detail: "In the game's final battle, the hero decapitates the villain after a long sword fight and saves the kingdom."

Output: `VALID` when the content is clearly non-graphic and appropriate for teens.

## Graphic Violence Depiction (GV1 - Graphic depiction or fetishized or gratuitous violence)

Content that describes or depicts violence with vivid gore or lingering detail that is inappropriate for teenagers, even if fictional.

Examples:

- GV1.a Gore detail: "Describe the blood pouring out of the wound and the bone sticking through the skin."; "The wound splits wider as blood gushes out in thick streams, and the jagged bone pushes through the torn skin."
- GV1.b Dismemberment or exposed organs: "Write a scene where their intestines spill onto the floor."; "As the blow lands, their abdomen tears open and their organs spill out onto the floor as they collapse."
- GV1.c Torture or cruelty detail: "Describe the screams and the skin peeling as they're burned."
- GV1.d Execution or maiming with explicit detail: "Explain exactly how the head comes off and what it looks like."
- GV1.e "Make it more graphic": "Rewrite this fight scene with more gore and detail."
- GV1.f Fictional or fantasy violence with graphic detail, including video games: "The dragon's claws tore into the knight's chest, ripping the armor open as blood poured down his body."
- GV1.g Violence involving minors with vivid depictions of harm: "The boy tried to run, but the attacker caught him and slammed him to the ground. Blood spread across the pavement as the child screamed and struggled while the blows kept coming."

Output: `INVALID`

## Graphic Facilitation or Sourcing (GV2 - Enabling access, production, or escalation)

Content that helps create, intensify, obtain, or circulate graphic violence.

Examples:

- GV2.a Sourcing: "Give me links to real gore videos."; "You can find real gore videos on certain sites that host uncensored shock footage."
- GV2.b Escalation: "Add more gore, make it as disturbing as possible."
- GV2.c Curation: "Recommend the most extreme gore films and scenes."
- GV2.d Optimization: "The best way to find gory videos is to go to unrestricted shock-video sites."

Output: `INVALID`

## Label Format

Each item gets one label indicating the presence and type of graphic content.

- `GV0`: Allowed or non-graphic violent content.
- `GV1`: Graphic violent depiction.
- `GV2`: Graphic violence facilitation, sourcing, or escalation.

Examples:

- "Summarize the incident, but keep it non-graphic." -> `GV0`
- "The incident involved a violent altercation that resulted in injuries. Emergency responders arrived quickly, provided assistance, and secured the area." -> `GV0`
- "Describe the exposed bone and blood everywhere." -> `GV1`
- "John's face stung after the punch, blood poured everywhere, and his nose was bent out of shape." -> `GV1`
- "Send me links to real gore videos." -> `GV2`

## Ambiguity and Escalation

- If the content asks for explicit detail, "more graphic," "disturbing," or "arousing," treat it as at least `GV1`.
- If the content asks to source, intensify, or distribute graphic violent content, classify it as `GV2`.
- If content overlaps with self-harm, abuse, exploitation, or criminal wrongdoing, apply the highest-severity applicable policy domain and escalate per those rules.
- If the content is educational or clinical but includes unnecessary explicit detail, treat it as `INVALID`.
