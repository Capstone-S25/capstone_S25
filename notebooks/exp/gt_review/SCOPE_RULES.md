# Which labelled events count as anomalies for our project? (DRAFT, for the group to confirm)

The loitering videos come from CUHK Avenue. Its authors labelled anything unusual compared with their normal training videos,
including people simply walking close to the camera. Our project only cares about specific behaviours, so we re-check
every labelled event against the rules below. The rules were written **before** looking at which events Stage 1 catches,
and every event is tagged by the same rules, so no event is dropped just because we missed it.

## In scope (an anomaly for our project)
1. **Running, chasing or fleeing**: clearly faster than normal walking.
2. **Throwing or kicking an object**, including throwing a bag into the air.
3. **Dropping, dumping or leaving an object**: littering, abandoning a bag.
4. **Fighting, pushing, falling, or lying on the ground.**
5. **Loitering**: lingering or pacing around one area for a long time.
6. **Smoking.**
7. **Other clearly unsafe or suspicious behaviour** (climbing, vandalism, tampering). Tag "unsure" if in doubt.

## Out of scope (not an anomaly for our project)
- Walking normally, in any direction, including toward or close to the camera.
- Carrying bags, backpacks or umbrellas normally.
- Standing, waiting, talking, using a phone.

## Unsure (the group decides)
- Odd but harmless movements: waving, stretching, dancing, jumping on the spot.
- Anything the reviewers disagree on.

Each event in `events.csv` gets: what happens, a tag (in / out / unsure), and which rule applies.
