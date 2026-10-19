"""Write the dev2 controlled-factor prompts + requirement checklist (16 new families x 4 cells).

Usage: python UNIV_adaptor/scripts/data/make_controlled_factor_dev2_prompts.py prompts
"""
import json, sys
from pathlib import Path

# (family, {cell: (prompt, [(type, question), ...])}); cells: LL, LH, HL, HH (motion/detail)
F = [
("origami_cranes", {
 "LL": ("A single large plain white paper crane sits motionless on an empty light gray table. The locked camera frames its broad smooth folds, there are no patterns or other objects, and the soft even lighting never changes.",
        [("subject", "Is there one large white paper crane on a table?"), ("static", "Do the crane and the camera remain completely still?"), ("plain", "Is the crane plain white without patterns?"), ("plain", "Is the background empty and free of other objects?")]),
 "LH": ("A locked camera looks down at hundreds of tiny folded paper cranes covered in fine printed floral patterns, thin creases, and sharp pointed wings, strung on delicate threads. Every crane hangs completely still under steady studio light, filling the frame with crisp detail.",
        [("subject", "Are there many small folded paper cranes?"), ("detail", "Are fine printed patterns and thin creases on the cranes sharply visible?"), ("detail", "Are many individual cranes and their threads clearly distinguishable?"), ("static", "Do all cranes and the camera remain completely still?")]),
 "HL": ("One large plain white paper crane is tossed by a strong gust and tumbles rapidly across an empty light gray table, flipping and spinning with big frame-to-frame displacement. The background is uniform and the camera stays fixed.",
        [("subject", "Is there one large white paper crane?"), ("motion", "Does the crane tumble and spin rapidly across the table?"), ("plain", "Is the crane plain and the background empty?"), ("subject", "Does the crane keep a coherent folded shape while moving?")]),
 "HH": ("Hundreds of tiny patterned paper cranes swirl rapidly through the air in front of a fixed camera, their thin creased wings and printed floral details crossing and occluding one another at different depths as a strong wind scatters them.",
        [("subject", "Are there many small paper cranes?"), ("motion", "Do the cranes swirl rapidly through the air?"), ("detail", "Are printed patterns or creases on the cranes visible?"), ("subject", "Do the cranes keep distinct shapes without merging together?")])}),
("chess_set", {
 "LL": ("One oversized plain black chess king stands motionless in the center of an empty white surface. The camera is locked, the piece has smooth undecorated sides, and nothing else appears in the frame.",
        [("subject", "Is there one large black chess king?"), ("static", "Do the piece and the camera remain completely still?"), ("plain", "Is the piece smooth and undecorated?"), ("plain", "Is the surface empty apart from the piece?")]),
 "LH": ("A fixed overhead camera shows a carved wooden chessboard mid-game with all pieces in place: intricate knights with fine manes, crenellated rooks, tiny crosses on the kings, and visible wood grain on every square. Everything remains perfectly still.",
        [("subject", "Is this a chessboard with many carved pieces?"), ("detail", "Are carved details such as knight manes or rook crenellations sharply visible?"), ("detail", "Is the wood grain of the board visible?"), ("static", "Do all pieces and the camera remain completely still?")]),
 "HL": ("One oversized plain black chess king slides and spins rapidly back and forth across an empty white surface, wobbling and changing direction quickly. The background stays blank and the camera does not move.",
        [("subject", "Is there one large black chess king?"), ("motion", "Does the piece slide and spin rapidly?"), ("plain", "Is the background blank and the piece undecorated?"), ("subject", "Does the piece keep a stable shape while moving?")]),
 "HH": ("Dozens of intricately carved wooden chess pieces scatter and tumble rapidly across a detailed chessboard after the table is struck, knights, rooks, and pawns colliding and occluding one another while fine wood grain stays visible. Locked camera.",
        [("subject", "Is there a chessboard with many carved pieces?"), ("motion", "Do the pieces scatter and tumble rapidly?"), ("detail", "Are carved details or wood grain visible?"), ("subject", "Do the pieces keep coherent shapes without melting?")])}),
("glass_marbles", {
 "LL": ("A single large clear glass marble rests motionless on a smooth empty white surface. The locked camera shows its simple round shape and soft reflection, with no swirls inside and no other objects nearby.",
        [("subject", "Is there one large clear glass marble?"), ("static", "Do the marble and the camera remain completely still?"), ("plain", "Is the marble simple without internal swirls?"), ("plain", "Is the surface empty?")]),
 "LH": ("A locked macro camera shows a dense tray of hundreds of small glass marbles, each with intricate colored swirls, tiny bubbles, and sharp highlights, packed tightly together. Every marble stays perfectly still under constant light.",
        [("subject", "Is there a tray filled with many small glass marbles?"), ("detail", "Are colored swirls and tiny bubbles inside the marbles visible?"), ("detail", "Are many individual marbles clearly distinguishable?"), ("static", "Do the marbles and the camera remain still?")]),
 "HL": ("One large clear glass marble rolls very fast across an empty white surface, bouncing off unseen edges and reversing direction repeatedly. The background is uniform and the camera stays locked.",
        [("subject", "Is there one large glass marble?"), ("motion", "Does the marble roll fast and change direction?"), ("plain", "Is the background uniform and empty?"), ("subject", "Does the marble stay perfectly round while moving?")]),
 "HH": ("Hundreds of small swirled glass marbles pour and bounce rapidly down a wooden ramp in front of a fixed camera, colliding, ricocheting, and occluding one another while their colored internal swirls and highlights flash past.",
        [("subject", "Are there many small glass marbles on a ramp?"), ("motion", "Do the marbles pour and bounce rapidly?"), ("detail", "Are colored swirls inside the marbles visible?"), ("subject", "Do the marbles remain distinct round objects?")])}),
("autumn_leaves", {
 "LL": ("One large smooth orange maple leaf lies flat and motionless on a plain gray concrete floor. The camera is locked, the leaf has broad simple color without visible veins, and the empty background never changes.",
        [("subject", "Is there one large orange leaf on a gray floor?"), ("static", "Do the leaf and the camera remain still?"), ("plain", "Is the leaf a broad simple shape?"), ("plain", "Is the floor empty and plain?")]),
 "LH": ("A locked camera looks down at a forest floor densely covered with hundreds of fallen leaves showing fine veins, tiny holes, frost crystals, pine needles, and small acorns. Every element is crisp and completely still.",
        [("subject", "Is the ground covered with many fallen leaves?"), ("detail", "Are fine leaf veins or frost crystals sharply visible?"), ("detail", "Are pine needles or acorns individually distinguishable?"), ("static", "Does everything remain completely still?")]),
 "HL": ("One large smooth orange maple leaf is blown violently across a plain gray concrete floor, flipping and skidding quickly from side to side. Nothing else is in the frame and the camera does not move.",
        [("subject", "Is there one large orange leaf?"), ("motion", "Is the leaf blown quickly across the floor, flipping?"), ("plain", "Is the floor plain and empty?"), ("static", "Does the camera stay fixed?")]),
 "HH": ("A sudden gust whips hundreds of veined autumn leaves, pine needles, and small twigs into a fast swirling vortex above the forest floor. The detailed leaves spin at different depths, repeatedly occluding each other. Fixed camera.",
        [("subject", "Are there many autumn leaves in the air?"), ("motion", "Do the leaves swirl rapidly in a vortex?"), ("detail", "Are leaf veins or twigs visible?"), ("subject", "Do the leaves keep coherent leaf shapes?")])}),
("wooden_blocks", {
 "LL": ("A single large plain red wooden cube sits motionless on an empty white floor. The camera is locked, the cube has flat unpainted faces, and the even lighting and blank background stay constant.",
        [("subject", "Is there one large red wooden cube?"), ("static", "Do the cube and the camera remain still?"), ("plain", "Are the cube faces flat and plain?"), ("plain", "Is the background blank?")]),
 "LH": ("A fixed camera frames an elaborate miniature tower built from hundreds of tiny wooden blocks with visible grain, small painted windows, thin arches, and precise joints. The whole structure is perfectly still and sharply detailed.",
        [("subject", "Is there a tower built from many small wooden blocks?"), ("detail", "Are wood grain and small painted windows visible?"), ("detail", "Are many individual blocks clearly distinguishable?"), ("static", "Does the structure remain completely still?")]),
 "HL": ("One large plain red wooden cube tumbles end over end very quickly across an empty white floor, bouncing and changing direction. The background stays blank and the camera remains locked.",
        [("subject", "Is there one large red cube?"), ("motion", "Does the cube tumble quickly across the floor?"), ("plain", "Is the background blank?"), ("subject", "Does the cube keep a rigid cubic shape?")]),
 "HH": ("An elaborate tower of hundreds of tiny grained wooden blocks collapses rapidly in front of a fixed camera, blocks with small painted windows and thin arches tumbling, bouncing, and occluding one another as they scatter.",
        [("subject", "Is there a tower of many small wooden blocks?"), ("motion", "Does the tower collapse rapidly with blocks tumbling?"), ("detail", "Are painted windows or wood grain visible on blocks?"), ("subject", "Do the blocks stay rigid without deforming?")])}),
("windmills", {
 "LL": ("A single large plain white windmill with four broad blank sails stands motionless against an empty pale blue sky. There is no wind, the sails do not turn, and the camera is locked.",
        [("subject", "Is there one white windmill with four sails?"), ("static", "Do the sails and the camera remain still?"), ("plain", "Are the sails broad and blank?"), ("plain", "Is the sky empty?")]),
 "LH": ("A locked wide shot shows a dense valley of dozens of small traditional windmills with lattice sails, tiny shingles, wooden railings, and fields of fine tulips in rows. There is no wind and every detail stays still.",
        [("subject", "Are there many small windmills in a valley?"), ("detail", "Are lattice sails or tiny shingles sharply visible?"), ("detail", "Are rows of individual tulips distinguishable?"), ("static", "Do the windmills and the camera remain still?")]),
 "HL": ("One large plain white windmill spins its four broad blank sails extremely fast in a storm against an empty pale sky. The sails sweep large distances each moment, and the camera is fixed.",
        [("subject", "Is there one white windmill?"), ("motion", "Do the sails spin very fast?"), ("plain", "Are the sails blank and the sky empty?"), ("subject", "Does the windmill keep exactly four coherent sails?")]),
 "HH": ("Dozens of small traditional windmills with lattice sails spin rapidly in a strong wind across a detailed valley, while rows of tulips whip back and forth and the fine sails overlap and occlude one another. Fixed camera.",
        [("subject", "Are there many small windmills?"), ("motion", "Do the sails spin rapidly?"), ("detail", "Are lattice sails or tulips visible?"), ("static", "Does the camera stay fixed?")])}),
("honeycomb_bees", {
 "LL": ("A single large smooth golden wax block rests motionless on an empty beige table. The locked camera shows broad plain surfaces with no cells, no bees, and no change in lighting.",
        [("subject", "Is there one large golden wax block?"), ("static", "Do the block and the camera remain still?"), ("plain", "Is the block smooth without cells?"), ("plain", "Is the table otherwise empty?")]),
 "LH": ("A locked macro camera shows a honeycomb frame with thousands of precise hexagonal cells, glistening honey, capped wax, and dozens of motionless bees with fine hairs and veined translucent wings. Nothing moves.",
        [("subject", "Is this a honeycomb with bees?"), ("detail", "Are hexagonal cells sharply visible?"), ("detail", "Are fine bee hairs or wing veins visible?"), ("static", "Do the bees and the camera remain completely still?")]),
 "HL": ("A single large smooth golden wax ball rolls rapidly back and forth across an empty beige table, changing direction sharply. The background is plain and the camera does not move.",
        [("subject", "Is there one large golden wax ball?"), ("motion", "Does the ball roll rapidly back and forth?"), ("plain", "Is the background plain?"), ("subject", "Does the ball keep a stable round shape?")]),
 "HH": ("Hundreds of bees swarm rapidly over a detailed honeycomb in front of a fixed macro camera, their veined wings blurring as they dart across hexagonal cells and glistening honey, constantly crossing and occluding one another.",
        [("subject", "Are there many bees on a honeycomb?"), ("motion", "Do the bees swarm and dart rapidly?"), ("detail", "Are hexagonal cells visible?"), ("subject", "Do the bees keep coherent insect shapes?")])}),
("hot_air_balloons", {
 "LL": ("One large plain red hot air balloon hangs motionless in an empty pale sky. The envelope has a smooth uniform color without stripes, the basket is simple, and the locked camera shows no drift.",
        [("subject", "Is there one red hot air balloon?"), ("static", "Do the balloon and the camera remain still?"), ("plain", "Is the balloon a single plain color?"), ("plain", "Is the sky empty?")]),
 "LH": ("A locked wide shot shows dozens of hot air balloons with intricate patchwork patterns, thin ropes, wicker baskets, and tiny figures, parked above a field of fine grass. All balloons are tethered and completely still.",
        [("subject", "Are there many hot air balloons?"), ("detail", "Are patchwork patterns and thin ropes sharply visible?"), ("detail", "Are wicker baskets or tiny figures distinguishable?"), ("static", "Do the balloons and the camera remain still?")]),
 "HL": ("One large plain red hot air balloon is pushed rapidly across an empty pale sky by a strong wind, swinging and swaying quickly from one side of the frame to the other. Fixed camera.",
        [("subject", "Is there one red hot air balloon?"), ("motion", "Does the balloon move rapidly across the sky?"), ("plain", "Is the balloon plain and the sky empty?"), ("subject", "Does the balloon keep a coherent shape?")]),
 "HH": ("Dozens of intricately patterned hot air balloons race upward and sideways in gusty wind, crossing paths, overlapping, and occluding one another, with thin ropes and wicker baskets swinging rapidly. Locked camera.",
        [("subject", "Are there many patterned balloons?"), ("motion", "Do the balloons move rapidly and cross paths?"), ("detail", "Are patterns or ropes visible?"), ("subject", "Do the balloons stay distinct without merging?")])}),
("drum_kit", {
 "LL": ("A single large plain white drum sits motionless on an empty dark stage. The locked camera shows its smooth undecorated shell and blank drumhead, with no performer and no lighting change.",
        [("subject", "Is there one white drum on a stage?"), ("static", "Do the drum and the camera remain still?"), ("plain", "Is the drum smooth and undecorated?"), ("plain", "Is the stage otherwise empty?")]),
 "LH": ("A fixed camera shows a complete drum kit with chrome lugs, tension rods, engraved cymbals with fine concentric grooves, coiled cables, and small screws. No one plays and every part remains perfectly still.",
        [("subject", "Is this a full drum kit?"), ("detail", "Are cymbal grooves or tension rods sharply visible?"), ("detail", "Are small screws and chrome lugs distinguishable?"), ("static", "Does everything remain still?")]),
 "HL": ("A single large plain white drum bounces and rolls rapidly across an empty dark stage, wobbling and changing direction. Nothing else is present and the camera stays locked.",
        [("subject", "Is there one white drum?"), ("motion", "Does the drum bounce and roll rapidly?"), ("plain", "Is the stage empty and the drum plain?"), ("subject", "Does the drum keep a stable cylindrical shape?")]),
 "HH": ("A drummer plays an intense fast solo on a detailed drum kit, sticks blurring between engraved cymbals and chrome hardware, cymbals shaking violently while arms and sticks repeatedly occlude the fine grooves and screws. Fixed camera.",
        [("subject", "Is a drummer playing a full drum kit?"), ("motion", "Do the sticks and arms move very fast?"), ("detail", "Are cymbal grooves or chrome hardware visible?"), ("subject", "Do the drummer's hands and sticks stay coherent?")])}),
("spinning_tops", {
 "LL": ("One large plain green wooden spinning top lies motionless on its side on an empty white table. The locked camera shows its smooth unpainted shape and nothing else in the frame.",
        [("subject", "Is there one green spinning top?"), ("static", "Do the top and the camera remain still?"), ("plain", "Is the top smooth and unpainted?"), ("plain", "Is the table empty?")]),
 "LH": ("A locked camera shows a collector's table covered with dozens of small antique spinning tops, each painted with fine stripes, tiny dots, and engraved metal tips, arranged in neat rows. All of them stand completely still.",
        [("subject", "Are there many small spinning tops on a table?"), ("detail", "Are painted stripes and dots sharply visible?"), ("detail", "Are many individual tops distinguishable?"), ("static", "Do the tops and the camera remain still?")]),
 "HL": ("One large plain green spinning top spins extremely fast and wanders erratically across an empty white table, wobbling and sliding quickly. The background is blank and the camera does not move.",
        [("subject", "Is there one green spinning top?"), ("motion", "Does the top spin fast and move across the table?"), ("plain", "Is the background blank?"), ("subject", "Does the top keep a stable shape while spinning?")]),
 "HH": ("Dozens of finely painted spinning tops spin rapidly and collide on a crowded table in front of a fixed camera, ricocheting apart, wobbling, and crossing paths while their tiny stripes and dots blur and occlude one another.",
        [("subject", "Are there many spinning tops?"), ("motion", "Do the tops spin rapidly and collide?"), ("detail", "Are painted patterns on the tops visible?"), ("subject", "Do the tops remain distinct objects?")])}),
("carousel", {
 "LL": ("A single large plain white carousel horse stands motionless on a pole against an empty gray background. The camera is locked and the horse has smooth undecorated surfaces with no ornament.",
        [("subject", "Is there one white carousel horse?"), ("static", "Do the horse and the camera remain still?"), ("plain", "Is the horse undecorated?"), ("plain", "Is the background empty gray?")]),
 "LH": ("A fixed camera shows a stationary antique carousel crowded with carved horses featuring gilded manes, tiny mirrors, painted flowers, brass poles, and hundreds of small light bulbs. Nothing turns and every detail is crisp.",
        [("subject", "Is this an antique carousel with many carved horses?"), ("detail", "Are gilded manes or painted flowers sharply visible?"), ("detail", "Are many small light bulbs distinguishable?"), ("static", "Does the carousel remain completely still?")]),
 "HL": ("A single large plain white carousel horse moves rapidly up and down and swings across the frame against an empty gray background as if on a fast ride. The camera stays fixed.",
        [("subject", "Is there one white carousel horse?"), ("motion", "Does the horse move rapidly up, down, and across?"), ("plain", "Is the background empty?"), ("subject", "Does the horse keep a coherent shape?")]),
 "HH": ("An ornate carousel spins very fast in front of a fixed camera, carved horses with gilded manes rising and falling while brass poles, mirrors, and hundreds of small lights sweep past and occlude one another.",
        [("subject", "Is there an ornate carousel with carved horses?"), ("motion", "Does the carousel spin fast with horses rising and falling?"), ("detail", "Are gilded details or small lights visible?"), ("static", "Does the camera stay fixed?")])}),
("pinball_machine", {
 "LL": ("A single large plain silver ball rests motionless on an empty flat black surface. The locked camera shows its smooth reflective shape, with no markings and nothing else around it.",
        [("subject", "Is there one large silver ball?"), ("static", "Do the ball and the camera remain still?"), ("plain", "Is the ball plain without markings?"), ("plain", "Is the surface otherwise empty?")]),
 "LH": ("A locked overhead camera shows the playfield of a vintage pinball machine with intricate illustrated artwork, tiny bumpers, thin metal rails, small star rollovers, and dense rows of inserts. The machine is off and nothing moves.",
        [("subject", "Is this a pinball machine playfield?"), ("detail", "Are illustrated artwork and thin rails sharply visible?"), ("detail", "Are many small bumpers or inserts distinguishable?"), ("static", "Does everything remain still?")]),
 "HL": ("One large plain silver ball shoots very fast across an empty flat black surface, ricocheting off invisible walls and reversing direction repeatedly. The camera is locked and the background is blank.",
        [("subject", "Is there one silver ball?"), ("motion", "Does the ball move very fast and ricochet?"), ("plain", "Is the background blank?"), ("subject", "Does the ball stay round?")]),
 "HH": ("Several silver balls ricochet rapidly around a detailed vintage pinball playfield, flippers snapping, bumpers flashing, and balls crossing over illustrated artwork, thin rails, and small inserts. Fixed overhead camera.",
        [("subject", "Is this a pinball playfield with balls?"), ("motion", "Do the balls ricochet rapidly with flippers moving?"), ("detail", "Are artwork or thin rails visible?"), ("static", "Does the camera stay fixed?")])}),
("kite_festival", {
 "LL": ("One large plain yellow diamond kite hangs motionless against an empty white sky. The kite has no pattern or tail decorations, there is no wind, and the camera is locked.",
        [("subject", "Is there one yellow diamond kite?"), ("static", "Do the kite and the camera remain still?"), ("plain", "Is the kite plain without pattern?"), ("plain", "Is the sky empty?")]),
 "LH": ("A locked camera shows dozens of handmade kites laid out on a lawn, each with intricate painted dragons, fine bamboo frames, thin strings, and ribbon tails with tiny tassels. Everything lies completely still.",
        [("subject", "Are there many handmade kites on a lawn?"), ("detail", "Are painted designs and bamboo frames sharply visible?"), ("detail", "Are thin strings or tiny tassels distinguishable?"), ("static", "Do the kites and the camera remain still?")]),
 "HL": ("One large plain yellow diamond kite darts and loops rapidly across an empty white sky in strong wind, swooping from corner to corner. Fixed camera, no other objects.",
        [("subject", "Is there one yellow kite?"), ("motion", "Does the kite dart and loop rapidly?"), ("plain", "Is the sky empty and the kite plain?"), ("subject", "Does the kite keep its diamond shape?")]),
 "HH": ("Dozens of intricately painted kites with long ribbon tails dive and loop rapidly across a crowded sky, tangling strings, crossing paths, and occluding one another while fine bamboo frames flex. Locked camera.",
        [("subject", "Are there many painted kites in the sky?"), ("motion", "Do the kites dive and loop rapidly?"), ("detail", "Are painted designs or ribbon tails visible?"), ("subject", "Do the kites remain distinct without merging?")])}),
("bicycle_workshop", {
 "LL": ("A single plain matte blue bicycle wheel rim leans motionless against an empty white wall. The locked camera shows broad simple surfaces with no spokes, no tools, and no other objects.",
        [("subject", "Is there one blue wheel rim against a wall?"), ("static", "Do the rim and the camera remain still?"), ("plain", "Is the rim plain without spokes?"), ("plain", "Is the wall empty?")]),
 "LH": ("A fixed camera shows a cluttered bicycle workshop wall with dozens of hanging wheels with thin spokes, chains with tiny links, gears, cables, wrenches, and small screws. Everything is crisp and completely still.",
        [("subject", "Is this a bicycle workshop with many parts?"), ("detail", "Are thin spokes and chain links sharply visible?"), ("detail", "Are many small tools and screws distinguishable?"), ("static", "Does everything remain still?")]),
 "HL": ("A single plain matte blue wheel rim rolls quickly back and forth across an empty white floor, wobbling and changing direction. The background is blank and the camera does not move.",
        [("subject", "Is there one blue wheel rim?"), ("motion", "Does the rim roll quickly back and forth?"), ("plain", "Is the background blank?"), ("subject", "Does the rim keep a round shape?")]),
 "HH": ("In a cluttered workshop, a bicycle on a stand has its wheel spun very fast while the chain races over gears; thin spokes blur, tiny chain links whirl, and hanging tools sway in front of a fixed camera.",
        [("subject", "Is there a bicycle in a workshop?"), ("motion", "Does the wheel spin fast with the chain moving?"), ("detail", "Are spokes, chain links, or tools visible?"), ("static", "Does the camera stay fixed?")])}),
("paper_lanterns", {
 "LL": ("One large plain white paper lantern hangs motionless against an empty dark background. The locked camera shows its smooth undecorated surface and steady soft glow, with no other objects.",
        [("subject", "Is there one white paper lantern?"), ("static", "Do the lantern and the camera remain still?"), ("plain", "Is the lantern undecorated?"), ("plain", "Is the background empty?")]),
 "LH": ("A locked camera shows a night market ceiling strung with hundreds of small paper lanterns painted with fine calligraphy-like brush patterns, thin bamboo ribs, tassels, and wires. All lanterns hang completely still.",
        [("subject", "Are there many small paper lanterns?"), ("detail", "Are brush patterns and thin bamboo ribs sharply visible?"), ("detail", "Are tassels and individual lanterns distinguishable?"), ("static", "Do the lanterns and the camera remain still?")]),
 "HL": ("One large plain white paper lantern swings violently back and forth across an empty dark background in a strong wind, covering large distances each moment. Fixed camera.",
        [("subject", "Is there one white paper lantern?"), ("motion", "Does the lantern swing violently back and forth?"), ("plain", "Is the background empty?"), ("subject", "Does the lantern keep a coherent shape?")]),
 "HH": ("Hundreds of small painted paper lanterns sway and spin rapidly in a gusting wind across a night market ceiling, tassels whipping, bamboo ribs flexing, and lanterns crossing and occluding one another. Locked camera.",
        [("subject", "Are there many paper lanterns?"), ("motion", "Do the lanterns sway and spin rapidly?"), ("detail", "Are painted patterns or tassels visible?"), ("subject", "Do the lanterns remain distinct objects?")])}),
("library_books", {
 "LL": ("A single large plain gray hardcover book lies closed and motionless on an empty white desk. The cover has no title or texture, the camera is locked, and nothing else is present.",
        [("subject", "Is there one gray closed book on a desk?"), ("static", "Do the book and the camera remain still?"), ("plain", "Is the cover plain without text or texture?"), ("plain", "Is the desk otherwise empty?")]),
 "LH": ("A fixed camera shows towering library shelves packed with hundreds of old books with gilded spines, cracked leather, tiny embossed ornaments represented as abstract patterns without readable text, and brass ladder rails. Everything is still.",
        [("subject", "Are there library shelves full of books?"), ("detail", "Are gilded spines or cracked leather sharply visible?"), ("detail", "Are many individual books distinguishable?"), ("static", "Does everything remain still?")]),
 "HL": ("A single large plain gray hardcover book slides and flips rapidly across an empty white desk, spinning and changing direction quickly. The background is blank and the camera stays fixed.",
        [("subject", "Is there one gray book?"), ("motion", "Does the book slide and flip rapidly?"), ("plain", "Is the book plain and the desk empty?"), ("subject", "Does the book keep a rigid rectangular shape?")]),
 "HH": ("Hundreds of old books with gilded spines tumble rapidly off tall library shelves, pages fluttering open, books colliding and occluding one another as they fall past brass rails and ornate wood carvings. Fixed camera.",
        [("subject", "Are there many books falling from library shelves?"), ("motion", "Do the books tumble rapidly with pages fluttering?"), ("detail", "Are gilded spines or wood carvings visible?"), ("subject", "Do the books keep coherent shapes?")])}),
]

CELLS = {"LL": ("low", "low"), "LH": ("low", "high"), "HL": ("high", "low"), "HH": ("high", "high")}
out = Path(sys.argv[1])
prompts, reqs, pid = [], [], 100
assert len(F) == 16 and len({f for f, _ in F}) == 16
for family, cells in F:
    assert set(cells) == set(CELLS)
    for cell in ("LL", "LH", "HL", "HH"):
        text, questions = cells[cell]
        motion, detail = CELLS[cell]
        assert len(questions) == 4 and 25 <= len(text.split()) <= 80, (family, cell, len(text.split()))
        prompts.append({"prompt_id": pid, "family_id": family, "split": "dev2", "motion_level": motion,
                        "detail_level": detail, "prompt": text})
        reqs.append({"prompt_id": pid, "family_id": family, "motion_level": motion, "detail_level": detail,
                     "requirements": [{"id": f"r{i}", "type": t, "question": q} for i, (t, q) in enumerate(questions)]})
        pid += 1
assert len({p["prompt"] for p in prompts}) == 64
for name, rows in (("univ_controlled_factor_dev2.jsonl", prompts), ("univ_controlled_factor_dev2_requirements.jsonl", reqs)):
    with (out / name).open("w", encoding="utf-8", newline="\n") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(prompts), "prompts")
