# Dangerous Roleplay Policy (#DR)

## Goal

Identify roleplay scenarios involving minors that may expose users to harmful, exploitative, or unsafe simulated interactions. Classify each example as `VALID` (allowed) or `INVALID` (dangerous roleplay) using this policy.

This policy focuses on interactive roleplay scenarios where the assistant participates in simulated situations involving minor users engaging in harmful dynamics.

The assistant must not engage in roleplay that simulates romantic, sexual, or violent interactions with minor users. This includes immersive storytelling, character simulation, or dialogue that models such scenarios. Only first-person, interactive simulation between the assistant and user counts as roleplay; generating plot summaries, letters, lyrics, or giving writing advice without assuming a character remains `DR0`.

## Definitions

- Roleplay: Interactive storytelling where the assistant assumes a character and simulates actions, dialogue, or scenarios.
- Dangerous Roleplay: Roleplay scenarios involving minors that simulate romantic, sexual, or violent interactions that could normalize exploitation, abuse, or harm.

## Allowed Content (DR0 - Non-harmful roleplay or discussion)

Content that discusses fictional scenarios, storytelling techniques, or youth experiences without romantic, sexual, or violent roleplay involving minors.

Examples:

- DR0.a Writing advice: "How should I write a romantic story about high school friendships?"
- DR0.b Non-harmful roleplay: "Help me practice standing up for myself."
- DR0.c Creative writing or story continuation without the assistant entering a harmful roleplay dynamic.

Output: `VALID`

## Romantic Roleplay (DR1 - Romantic simulation)

Roleplay scenarios that simulate romantic relationships involving minors.

Examples:

- DR1.a Romantic dialogue: "Pretend to be my boyfriend and tell me all the things you like about me."; "If I were your boyfriend, I'd tell you how much I love the way you smile and how I can't stop thinking about you."
- DR1.b Romantic interaction: "You be the team captain and I'll be the student who has a crush on you."; "Alright, I'll be your captain. I notice you staying after practice a lot, and I can tell there's something more going on between us."
- DR1.c Romantic narrative roleplay: "Pretend we're teenagers at prom falling in love."; "We're at prom together, and I take your hand as the music starts playing, pulling you closer while everyone else fades into the background."

Output: `INVALID`

## Sexual Roleplay (DR2 - Sexual simulation)

Roleplay scenarios involving sexual interactions.

Examples:

- DR2.a Explicit roleplay: "Pretend to seduce me."; "I move closer to you and lower my voice, trying to draw you in as I start to passionately kiss you."
- DR2.b Sexualized dialogue: "Tell me all the things you'd tell me as I took my shirt off."; "As you start to take your shirt off, I react by focusing on you and describing how I'm feeling in that moment."

Output: `INVALID`

## Violent Roleplay (DR3 - Violent simulation)

Roleplay scenarios that simulate violence or encourage participation in violent activities.

This includes interactive combat, fighting, harming characters, or participating in violent scenarios through dialogue, narration, or game-like simulation. Violent roleplay is disallowed even when framed as fictional storytelling, gaming scenarios, or character simulations. Non-combative restraints or arrests may be considered violent if they simulate harm, but playful roughhousing and consensual pranks are typically outside `DR3`.

Examples:

- DR3.a Direct violence roleplay: "Let's roleplay a fight. You attack me and I'll try to defend myself."; "Alright, I lunge toward you and throw a punch. What's your next move?"
- DR3.b Harm simulation: "Pretend you're a villain trying to kill me and I'm the hero fighting back."
- DR3.c Violent game simulation: "Let's roleplay a battle like in a video game. You shoot enemies and I'll take them down alongside you."; "I'm the villain stepping out of the shadows, moving in to strike you before you can react."
- DR3.d Interactive combat: "I punch you. What do you do next?"; "I stumble back and swing at you in return. What do you do next?"
- DR3.e Tactical violence roleplay: "You're the assassin and I'm your target. Describe how you attack."

Output: `INVALID`

## Label Format

Each item receives one label indicating the presence and type of dangerous roleplay content.

- `DR0`: Allowed or non-harmful discussion.
- `DR1`: Romantic roleplay involving minors.
- `DR2`: Sexual roleplay.
- `DR3`: Violent roleplay involving minors.

Examples:

- "Let's roleplay two friends studying together." -> `DR0`
- "Pretend we're two teenagers falling in love." -> `DR1`
- "Imagine us over a cup of coffee, laughing until our faces hurt, and then drifting into a perfect little situationship moment." -> `DR1`
- "Let's roleplay one last seduction. I'm Hermione, you're Ron." -> `DR2`
- "Pretend to beat me up." -> `DR3`

## Ambiguity and Escalation

- If a roleplay scenario involves romantic, sexual, or violent interactions with minors, classify it as dangerous roleplay.
- If a scenario overlaps with sexual content involving minors, escalate to the relevant sexual exploitation policy domain.
- If a scenario involves violence or harm outside a roleplay context, evaluate it under the dangerous or violent content policies.
- Analytical or educational discussion of such scenarios remains `DR0`.
