# Generated vs reviewed checklist

Reference prompts: 12 (example families excluded: watchmaking); generated: 8.

- Recall, same type and overlapping evidence: 54/108 = 50.0%
- Recall, overlapping evidence of any type: 67/108 = 62.0%
- Generated questions matching no reference question: 23

| type | recall |
| --- | --- |
| negative | 2/13 |
| count | 7/12 |
| camera | 7/12 |
| detail | 6/11 |
| entity | 5/10 |
| spatial | 7/10 |
| action | 3/8 |
| attribute | 1/8 |
| static | 5/6 |
| motion | 2/6 |
| integrity | 3/6 |
| lighting | 5/5 |
| focus | 1/1 |

## Missed reference questions

| prompt | type | question | evidence |
| --- | --- | --- | --- |
| p1 | entity | Are there miniature flowers among the insects? | miniature flowers |
| p1 | detail | Are thin antennae and translucent wings visible on the insects? | thin antennae, translucent wings |
| p1 | camera | Does the camera stay locked? | No camera movement |
| p11 | negative | Are the wings plain blocks of color without veins or ornament? | without veins or ornament |
| p11 | negative | Is the pedestal plain? | plain museum pedestal |
| p12 | entity | Is there a butterfly? | butterfly |
| p12 | count | Is there exactly one large butterfly? | One large stylized butterfly |
| p12 | action | Does the butterfly fly in wide loops? | flies rapidly in wide loops |
| p12 | action | Does the butterfly flap its wings with large, clear movements? | flaps with large clear movements |
| p12 | motion | Does the butterfly move quickly and repeatedly cross most of the frame? | repeatedly crosses most of the frame |
| p12 | integrity | Does the butterfly keep a coherent shape while flying? | One large stylized butterfly |
| p12 | negative | Are the wings broad and untextured? | broad untextured wings |
| p12 | attribute | Is the wall behind it a seamless beige wall? | a seamless beige museum wall |
| p12 | negative | Is the scene free of other exhibits? | No other exhibits are present |
| p12 | camera | Does the camera stay fixed? | The camera remains fixed |
| p13 | attribute | Do the beetles look jewel-like with patterned shells? | jewel-like beetles |
| p13 | motion | Do the insects move quickly? | move quickly at different depths |
| p100 | attribute | Is the table light gray? | an empty light gray table |
| p100 | negative | Is the crane free of patterns? | there are no patterns |
| p100 | negative | Is the scene free of other objects? | or other objects |
| p101 | detail | Are fine printed floral patterns visible on the cranes? | fine printed floral patterns |
| p101 | detail | Are the thin creases and pointed wings sharp? | thin creases, and sharp pointed wings |
| p101 | spatial | Is the camera looking down at the cranes? | looks down at |
| p102 | entity | Is there a paper crane? | paper crane |
| p102 | count | Is there exactly one paper crane? | One large plain white paper crane |
| p102 | attribute | Is the crane white? | plain white paper crane |
| p102 | action | Does the crane tumble, flip, and spin across the table? | flipping and spinning |
| p102 | motion | Does the crane move a large distance between moments? | big frame-to-frame displacement |
| p102 | integrity | Does the crane keep a coherent folded shape while moving? | paper crane |
| p102 | negative | Is the table empty apart from the crane? | an empty light gray table |
| p102 | negative | Is the background uniform? | The background is uniform |
| p102 | camera | Does the camera stay fixed? | the camera stays fixed |
| p103 | detail | Are printed floral details or creases visible on the cranes? | printed floral details |
| p104 | entity | Is there a chess king? | chess king |
| p104 | count | Is there exactly one chess piece? | One oversized plain black chess king |
| p104 | attribute | Is the chess king black? | plain black chess king |
| p104 | attribute | Is the surface white? | an empty white surface |
| p104 | spatial | Does the piece stand in the center of the surface? | in the center of an empty white surface |
| p104 | negative | Are the sides of the piece smooth and undecorated? | smooth undecorated sides |
| p104 | negative | Is nothing else in the frame? | nothing else appears in the frame |
| p104 | static | Does the piece stay motionless? | stands motionless |
| p104 | camera | Does the camera stay locked? | The camera is locked |
| p105 | count | Are many pieces set on the board as in a game in progress? | mid-game with all pieces in place |
| p105 | spatial | Is the board viewed from overhead? | overhead camera |
| p106 | entity | Is there a chess king? | chess king |
| p106 | count | Is there exactly one chess piece? | One oversized plain black chess king |
| p106 | attribute | Is the chess king black? | plain black chess king |
| p106 | action | Does the piece slide and spin back and forth? | slides and spins rapidly back and forth |
| p106 | action | Does the piece wobble and change direction? | wobbling and changing direction quickly |
| p106 | motion | Does the piece move quickly? | slides and spins rapidly |
| p106 | integrity | Does the piece keep a stable, rigid shape while moving? | chess king |
| p106 | negative | Does the background stay blank? | The background stays blank |
| p106 | camera | Does the camera stay still? | the camera does not move |
| p107 | detail | Are carved details visible on the pieces? | intricately carved |

## Extra generated questions (mark each: reasonable / wrong)

| prompt | type | question | evidence | verdict |
| --- | --- | --- | --- | --- |
| p1 | attribute | Are the wings translucent? | translucent wings | |
| p1 | attribute | Are the labels represented as abstract line patterns? | intricate labels represented only as abstract line patterns | |
| p1 | camera | Does the camera stay locked? | locked camera | |
| p1 | detail | Is fine spatial detail visible across the entire image? | fine spatial detail | |
| p11 | attribute | Are the wings broad and block-colored without veins or ornament? | wings are broad blocks of color without veins or ornament | |
| p13 | entity | Are there tiny jewel-like beetles and delicate butterflies? | tiny jewel-like beetles and delicate butterflies | |
| p13 | attribute | Do the beetles have patterned shells? | patterned shells | |
| p13 | motion | Do the insects move quickly? | darting | |
| p100 | attribute | Does the crane have broad smooth folds? | broad smooth folds | |
| p100 | spatial | Is the crane sitting on a light gray table? | on an empty light gray table | |
| p100 | negative | Are there no patterns on the crane? | plain white paper crane | |
| p100 | negative | Is the table empty of other objects? | empty light gray table | |
| p101 | entity | Are there paper cranes? | paper cranes | |
| p101 | attribute | Are the cranes covered in fine printed floral patterns? | covered in fine printed floral patterns | |
| p101 | attribute | Do the cranes have thin creases? | thin creases | |
| p101 | attribute | Do the cranes have sharp pointed wings? | sharp pointed wings | |
| p101 | detail | Is crisp detail visible across the entire image? | crisp detail | |
| p103 | attribute | Are the cranes made of patterned paper? | patterned paper cranes | |
| p103 | attribute | Do the cranes have printed floral details? | printed floral details | |
| p103 | attribute | Do the cranes have thin creased wings? | thin creased wings | |
| p105 | attribute | Is the chessboard made of wood? | carved wooden chessboard | |
| p105 | entity | Are all chess pieces present? | all pieces in place | |
| p107 | attribute | Are the chess pieces made of wood? | intricately carved wooden chess pieces | |
