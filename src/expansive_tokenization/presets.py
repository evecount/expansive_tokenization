"""
Curated Context-of-Interest Tokenization Sets & Polysemy Contrast Corpora.

Includes 6 foundational packs:
1. greek_mythology_poetry: Classical epic poetry, Homeric epithets & mythic concepts
2. ml_vs_general: Interlocutor adaptation (General everyday speaker vs. Machine Learning researcher)
3. generational_slang: Millennial vs. Gen-Z lexical sets & semantic drift
4. commonwealth_englishes: Singapore/Malaysia, UK, Australia, Canada vs. US English nuances
5. francophone_diatopic: Québécois French vs. Parisian/Verlan vs. West African Francophonie
6. culinary_turkey_dinner: Sensory & compositional scene tokenization ("The Turkey Dinner Experiment")
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass
class ContrastPair:
    """A shared target word placed in two distinct contexts of interest to measure 768-D embedding shift."""
    target_word: str
    context_a_label: str
    sentence_a: str
    context_b_label: str
    sentence_b: str
    semantic_note: str


@dataclass
class TokenizationSet:
    """A structured tokenization set tailored to a specific context of interest, dialect, or persona."""
    pack_id: str
    title: str
    persona_or_domain: str
    description: str
    ontological_categories: Dict[str, List[str]]
    sample_corpus: List[str]
    contrast_pairs: List[ContrastPair] = field(default_factory=list)
    semantic_definitions: Dict[str, str] = field(default_factory=dict)

    @property
    def all_tokens(self) -> List[str]:
        tokens: List[str] = []
        for cat_tokens in self.ontological_categories.values():
            for t in cat_tokens:
                if t not in tokens:
                    tokens.append(t)
        return tokens

    def to_dict(self) -> dict:
        return asdict(self)


CONTEXT_PACKS: Dict[str, TokenizationSet] = {
    "greek_mythology_poetry": TokenizationSet(
        pack_id="greek_mythology_poetry",
        title="Greek Mythology & Epic Poetry",
        persona_or_domain="Classical Philologist / Mythic Poet",
        description=(
            "Preserves Greek epic concepts, Homeric compound epithets, and prosodic structures as atomic tokens "
            "instead of shattering them into unrelated subword fragments."
        ),
        ontological_categories={
            "<|ETHOS_AND_FATE|>": [
                "kleos", "nostos", "xenia", "hubris", "nemesis", "moirai", "aristeia", "arete", "ate"
            ],
            "<|COSMOLOGY_AND_RITE|>": [
                "katabasis", "apotheosis", "chthonic", "ambrosia", "ichor", "aegis", "omphalos", "asphodel", "stygian"
            ],
            "<|POETIC_FORM_AND_EPITHET|>": [
                "rhapsode", "dactylic_hexameter", "caesura", "wine-dark_sea", "rosy-fingered_dawn", "swift-footed", "grey-eyed"
            ],
        },
        semantic_definitions={
            "katabasis": "descent into the underworld to retrieve forbidden wisdom",
            "kleos": "immortal renown earned through heroic song and valor",
            "nostos": "hard-won homecoming voyage across the sea",
            "xenia": "sacred law of hospitality binding host and stranger",
            "aristeia": "a hero's finest moment of unstoppable excellence in battle",
            "chthonic": "belonging to the subterranean deities of the dark earth",
            "rhapsode": "wandering epic bard who chants Homeric verse",
            "wine-dark_sea": "Homeric epithet for the deep, storm-tossed Aegean waters",
            "rosy-fingered_dawn": "Homeric epithet for the early morning light breaking over the horizon",
        },
        sample_corpus=[
            "The rhapsode chanted in dactylic_hexameter of Odysseus yearning for nostos across the wine-dark_sea.",
            "Driven by hubris, the hero risked a perilous katabasis into the chthonic realm of asphodel and stygian shadow.",
            "Under the sacred law of xenia, violating a guest's trust invites the wrath of nemesis and the moirai.",
            "At rosy-fingered_dawn, grey-eyed Athena raised her aegis to grant Achilles his final aristeia and undying kleos.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="shade",
                context_a_label="Everyday Modern English",
                sentence_a="We sat under the cool shade of the oak tree to escape the afternoon sun.",
                context_b_label="Homeric Underworld / Katabasis",
                sentence_b="Down in the House of Hades, the weeping shade of Achilles longed for a single breath of mortal life.",
                semantic_note="Shifts from physical shadow/shelter to a disembodied soul in the Underworld.",
            ),
            ContrastPair(
                target_word="weaver",
                context_a_label="Industrial Textile Trade",
                sentence_a="The factory weaver inspected the cotton loom for broken threads.",
                context_b_label="Mythic Fate & Guile (Penelope / Moirai)",
                sentence_b="By torchlight, the cunning weaver unravelled the burial shroud to delay her suitors and outwit fate.",
                semantic_note="Shifts from literal textile craftsmanship to mythic agency, stratagem, and destiny.",
            ),
        ],
    ),
    "ml_vs_general": TokenizationSet(
        pack_id="ml_vs_general",
        title="Interlocutor Adaptation: General Speaker vs. ML Researcher",
        persona_or_domain="Machine Learning Engineer / AI Researcher",
        description=(
            "Demonstrates how knowing your interlocutor is studying Machine Learning expands the tokenization set "
            "and radically re-orients the embeddings of everyday words like 'attention', 'weight', 'loss', and 'token'."
        ),
        ontological_categories={
            "<|ARCHITECTURE_PRIMITIVES|>": [
                "attention_head", "self-attention", "token_embedding", "positional_encoding",
                "residual_stream", "layer_norm", "kv_cache", "softmax_temperature"
            ],
            "<|OPTIMIZATION_AND_TRAINING|>": [
                "gradient_descent", "backpropagation", "cross_entropy_loss", "learning_rate_warmup",
                "weight_decay", "lora_adapter", "quantization", "perplexity"
            ],
            "<|TOKENIZATION_THEORY|>": [
                "byte_pair_encoding", "subword_fertility", "out_of_vocabulary", "special_control_token", "context_window"
            ],
        },
        semantic_definitions={
            "attention_head": "parallel subspace projection matrix computing token-to-token relevance weights",
            "token_embedding": "high-dimensional continuous vector representation encoding a token's semantic meaning",
            "gradient_descent": "iterative first-order optimization algorithm minimizing loss along parameter gradients",
            "subword_fertility": "average number of subword tokens required to encode a single natural word",
            "kv_cache": "memory buffer storing key and value tensors across autoregressive generation steps",
            "lora_adapter": "low-rank matrix decomposition module for parameter-efficient fine-tuning",
        },
        sample_corpus=[
            "During backpropagation, gradient_descent updates each attention_head and token_embedding to minimize cross_entropy_loss.",
            "High subword_fertility fragments rare domain terms, wasting the context_window and inflating kv_cache memory.",
            "We attached a quantized lora_adapter to fine-tune the residual_stream without shifting base model perplexity.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="attention",
                context_a_label="General Everyday Speaker",
                sentence_a="Please pay attention to the teacher while she explains the rules to the class.",
                context_b_label="Machine Learning Researcher",
                sentence_b="Multi-head causal attention computes scaled dot-product weights between query, key, and value matrices.",
                semantic_note="Shifts from human behavioral focus to bilinear matrix operations over token vectors.",
            ),
            ContrastPair(
                target_word="weights",
                context_a_label="General Everyday Speaker",
                sentence_a="He lifted heavy iron weights at the gym to build muscle strength.",
                context_b_label="Machine Learning Researcher",
                sentence_b="We froze the pretrained transformer weights and updated only the low-rank projection matrices.",
                semantic_note="Shifts from physical gym equipment to learnable floating-point parameters in a neural network.",
            ),
            ContrastPair(
                target_word="loss",
                context_a_label="General Everyday Speaker",
                sentence_a="She felt a deep sense of loss after moving away from her childhood hometown.",
                context_b_label="Machine Learning Researcher",
                sentence_b="The validation loss plateaued after three epochs of stochastic gradient descent.",
                semantic_note="Shifts from emotional grief to a scalar objective function measuring prediction error.",
            ),
        ],
    ),
    "generational_slang": TokenizationSet(
        pack_id="generational_slang",
        title="Generational Slang: Millennial vs. Gen-Z Semantic Drift",
        persona_or_domain="Sociolinguist / Generational Culture Analyst",
        description=(
            "Captures how generational cohorts coin atomic multi-word idioms and repurpose existing English words "
            "('cooked', 'bet', 'ate', 'aura') with sharply divergent semantic polarities."
        ),
        ontological_categories={
            "<|MILLENNIAL_LEXICON|>": [
                "adulting", "doggo", "pupper", "on_fleek", "side_hustle", "FOMO", "avocado_toast", "I_cant_even", "epic_fail"
            ],
            "<|GEN_Z_LEXICON|>": [
                "no_cap", "rizz", "skibidi", "delulu", "aura_points", "main_character_energy",
                "ate_and_left_no_crumbs", "brainrot", "let_him_cook", "it's_giving"
            ],
            "<|POLYSEMOUS_PIVOT_TOKENS|>": [
                "cooked", "bet", "tea", "cap", "slay", "based", "ghosted", "vibe_check"
            ],
        },
        semantic_definitions={
            "adulting": "performing mundane grown-up responsibilities like paying bills and scheduling appointments",
            "no_cap": "truthfully and without exaggeration or falsehood",
            "rizz": "effortless charisma and charm in interpersonal attraction",
            "delulu": "unrealistic or delusional optimism maintained as a playful mindset",
            "ate_and_left_no_crumbs": "executed a performance or style flawlessly with complete mastery",
            "let_him_cook": "allow someone to continue developing their idea or performance without interruption",
            "aura_points": "social coolness and composed presence gained or lost through public actions",
        },
        sample_corpus=[
            "Millennial context: After eight hours of exhausting adulting and my side_hustle, I have serious FOMO.",
            "Gen-Z context: No_cap, her presentation ate_and_left_no_crumbs—pure main_character_energy and infinite aura_points.",
            "Wait, let_him_cook! Everybody thought the strategy was delulu, but the rizz carried the entire pitch.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="cooked",
                context_a_label="Literal Culinary / Millennial Standard",
                sentence_a="My grandmother cooked a warm pot of chicken soup on the stove for dinner.",
                context_b_label="Gen-Z Slang (Doomed vs. Masterful)",
                sentence_b="I didn't study for the exam at all, bro I am completely cooked.",
                semantic_note="Shifts from preparing food with heat to being utterly defeated, exhausted, or doomed.",
            ),
            ContrastPair(
                target_word="bet",
                context_a_label="Traditional Gambling / Wager",
                sentence_a="He placed a twenty dollar bet on the horse race at the track.",
                context_b_label="Gen-Z Affirmative Discourse Marker",
                sentence_b="You want to grab boba at five o'clock? Bet, I will see you there.",
                semantic_note="Shifts from a risky monetary wager to an enthusiastic marker of agreement and confirmation.",
            ),
            ContrastPair(
                target_word="ate",
                context_a_label="Literal Consumption",
                sentence_a="We ate sandwiches and fresh fruit at the picnic table.",
                context_b_label="Gen-Z Aesthetic Praise",
                sentence_b="She walked onto the runway in that vintage gown and honestly she ate.",
                semantic_note="Shifts from ingesting food to delivering an extraordinary, flawless artistic or stylistic triumph.",
            ),
        ],
    ),
    "commonwealth_englishes": TokenizationSet(
        pack_id="commonwealth_englishes",
        title="World Englishes & Commonwealth Lexical Sets",
        persona_or_domain="Commonwealth Cross-Cultural Communicator (SG / UK / AU / CA)",
        description=(
            "Prevents non-US English varieties from being treated as noisy anomalies by equipping the tokenizer "
            "with Singaporean (Singlish), British, Australian, and Canadian lexical anchors."
        ),
        ontological_categories={
            "<|SINGAPORE_MALAYSIA_ENGLISH|>": [
                "shiok", "kiasu", "lepak", "makan", "tapau", "kopitiam", "paiseh", "bojio", "sian", "can_or_not"
            ],
            "<|UK_AND_HIBERNO_ENGLISH|>": [
                "fortnight", "chuffed", "knackered", "cheeky_nandos", "gobsmacked", "kerfuffle", "brolly", "queueing"
            ],
            "<|AUSTRALIAN_AND_CANADIAN_ENGLISH|>": [
                "arvo", "fair_dinkum", "strewth", "maccas", "toque", "double-double", "loonie", "washroom"
            ],
        },
        semantic_definitions={
            "shiok": "intensely delicious, satisfying, and pleasurable sensation",
            "kiasu": "driven by an anxious fear of missing out or falling behind competitors",
            "lepak": "relaxing casually and hanging out without urgency",
            "tapau": "packaging cooked food from a hawker stall to take away",
            "chuffed": "deeply pleased, proud, and delighted with an outcome",
            "knackered": "completely exhausted after long physical or mental effort",
            "arvo": "the afternoon hours of the day",
            "toque": "knitted winter beanie hat worn in cold Canadian weather",
        },
        sample_corpus=[
            "Singapore context: The laksa at the kopitiam was so shiok! Let's tapau two bowls before the kiasu crowd arrives.",
            "UK/AU context: I was proper chuffed after a fortnight of work, though completely knackered by Friday arvo.",
            "Canadian context: Grab your warm toque and a loonie, we're stopping for a double-double on the way.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="boot",
                context_a_label="North American English (Footwear)",
                sentence_a="He laced up his leather hiking boot before walking through the snow.",
                context_b_label="British & Commonwealth English (Vehicle Storage)",
                sentence_b="Put the suitcases and groceries into the boot of the car before we drive onto the motorway.",
                semantic_note="Shifts from heavy footwear to the rear luggage compartment (trunk) of an automobile.",
            ),
            ContrastPair(
                target_word="scheme",
                context_a_label="North American English (Deceptive Plot)",
                sentence_a="The con artist was arrested for running a fraudulent pyramid scheme.",
                context_b_label="UK & Singapore Public Policy English (Official Program)",
                sentence_b="The ministry launched a new housing grant scheme to support first-time young families.",
                semantic_note="Shifts from a sinister conspiracy to a legitimate, structured public initiative or government program.",
            ),
        ],
    ),
    "francophone_diatopic": TokenizationSet(
        pack_id="francophone_diatopic",
        title="Francophone Diatopic Nuances: Québec vs. Paris vs. Francophonie",
        persona_or_domain="Francophone Dialectologist (Québécois / Parisian / West African French)",
        description=(
            "Illustrates how a single language (French) diverges across Québec, Parisian registers (including Verlan), "
            "and the broader Francophonie—where shared words like 'char', 'chum', or 'dépanneur' carry completely distinct meanings."
        ),
        ontological_categories={
            "<|QUEBECOIS_LEXICON|>": [
                "dépanneur", "tiguidou", "poudrerie", "magasiner", "jasette", "tabarnak", "tuque", "breuvage", "fin_de_semaine"
            ],
            "<|PARISIAN_AND_VERLAN_REGISTER|>": [
                "meuf", "chelou", "relou", "boulot", "bagnole", "kiffer", "ouf", "bouquin"
            ],
            "<|PAN_FRANCOPHONE_AND_AFRICAN|>": [
                "teranga", "griot", "maquis", "palabre", "ambianceur", "essencerie"
            ],
        },
        semantic_definitions={
            "dépanneur": "neighborhood convenience store in Québec open late for daily essentials",
            "tiguidou": "everything is in great order, agreed, and going smoothly",
            "poudrerie": "blowing snow swirled into blinding clouds by strong winter gusts",
            "magasiner": "going shopping at retail stores",
            "chelou": "suspicious, strange, or shady in Parisian verlan slang",
            "teranga": "warm hospitality and generous welcoming spirit in Senegalese Francophone culture",
            "maquis": "lively neighborhood open-air restaurant and social gathering spot in West Africa",
        },
        sample_corpus=[
            "Québec context: J'ai pris mon char dans la poudrerie pour aller au dépanneur et magasiner avec mon chum.",
            "Paris context: Ce boulot est vraiment chelou, j'ai garé ma bagnole près du métro après avoir lu un bouquin.",
            "Dakar/Abidjan context: Le griot raconte l'histoire au maquis avec toute la chaleur de la teranga.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="char",
                context_a_label="Metropolitan Parisian French (Military Tank / Chariot)",
                sentence_a="Le char d'assaut blindé a traversé le terrain militaire pendant le défilé.",
                context_b_label="Québécois French (Everyday Automobile / Car)",
                sentence_b="J'ai stationné mon char devant le dépanneur pour acheter du lait.",
                semantic_note="In Paris, 'char' evokes an armored tank or parade float; in Québec, it is the everyday word for one's car.",
            ),
            ContrastPair(
                target_word="dépanneur",
                context_a_label="Metropolitan Parisian French (Repair Technician / Tow Truck)",
                sentence_a="Ma voiture est tombée en panne sur l'autoroute, j'ai dû appeler un dépanneur.",
                context_b_label="Québécois French (Corner Convenience Store)",
                sentence_b="On marche jusqu'au dépanneur du coin pour chercher des chips et une bière.",
                semantic_note="In France, a 'dépanneur' is an emergency repairman; in Québec, it is the neighborhood convenience store.",
            ),
        ],
    ),
    "culinary_turkey_dinner": TokenizationSet(
        pack_id="culinary_turkey_dinner",
        title="Sensory & Culinary Scene Decomposition ('The Turkey Dinner Experiment')",
        persona_or_domain="Culinary Ontologist & Scene Composer",
        description=(
            "Inspired by the question: 'How do you tokenize a turkey dinner?' Instead of arbitrary character chunks, "
            "we tokenize a scene by its functional & sensory roles: centrepiece protein, starches, tart/umami accents, "
            "thermal transformation vessels (oven), and plating artifacts."
        ),
        ontological_categories={
            "<|CENTERPIECE_AND_STARCHES|>": [
                "roast_turkey", "mashed_potatoes", "roasted_pumpkin", "herb_stuffing", "sweet_potato_casserole"
            ],
            "<|SAUCES_AND_CONTRAST_ACCENTS|>": [
                "pan_gravy", "cranberry_sauce", "sage_butter", "caramelized_shallots", "maillard_crust"
            ],
            "<|THERMAL_VESSELS_AND_TABLEWARE|>": [
                "convection_oven", "roasting_pan", "basting_bulb", "warmed_ceramic_plates", "carving_platter", "gravy_boat"
            ],
        },
        semantic_definitions={
            "roast_turkey": "golden-brown whole roasted poultry centerpiece with crisp skin and tender meat",
            "mashed_potatoes": "creamy whipped potatoes enriched with butter and warm milk",
            "pan_gravy": "savory emulsified sauce reduced from rich roasting pan drippings and stock",
            "cranberry_sauce": "bright tart-sweet ruby fruit compote that cuts through rich savory fats",
            "maillard_crust": "deeply savory browned surface formed by amino acid and sugar reactions at high oven heat",
            "warmed_ceramic_plates": "pre-heated dining plates that preserve the thermal aroma of the meal at service",
        },
        sample_corpus=[
            "Out of the convection_oven came the roast_turkey with a crackling maillard_crust inside the heavy roasting_pan.",
            "We arranged mashed_potatoes, roasted_pumpkin, pan_gravy, and tart cranberry_sauce across the warmed_ceramic_plates.",
            "Pouring pan_gravy from the gravy_boat ties the roast_turkey and mashed_potatoes into a single harmonious bite.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="dressing",
                context_a_label="Medical First Aid",
                sentence_a="The nurse applied a sterile gauze dressing to protect the healing wound.",
                context_b_label="Holiday Culinary Table",
                sentence_b="We spooned savory sage and cornbread dressing beside the roast turkey and gravy.",
                semantic_note="Shifts from sterile medical wound care to an aromatic herb bread side dish.",
            ),
            ContrastPair(
                target_word="plate",
                context_a_label="Tectonic Geology",
                sentence_a="The oceanic tectonic plate subducted beneath the continental crust, triggering a tremor.",
                context_b_label="Culinary Service & Plating",
                sentence_b="The chef arranged the turkey, potatoes, and cranberry reduction on a warm ceramic plate.",
                semantic_note="Shifts from planetary lithospheric slabs to tableware vessel design.",
            ),
        ],
    ),
    "iberian_lusophone_germanic": TokenizationSet(
        pack_id="iberian_lusophone_germanic",
        title="Iberian, Lusophone & Germanic Regional Continuums (PT-PT vs. PT-BR, Castilian vs. Catalan, Regional UK)",
        persona_or_domain="Comparative Romance & Germanic Dialectologist",
        description=(
            "Explores how closely related linguistic varieties—European vs. Brazilian Portuguese, Castilian Spanish vs. Catalan, "
            "and regional British English tonalities (Geordie, Scouse, Yorkshire, Cockney)—use distinct lexical tokens and assign "
            "divergent meanings to identical surface words."
        ),
        ontological_categories={
            "<|PORTUGAL_VS_BRAZIL_LUSOPHONE|>": [
                "autocarro", "ônibus", "pequeno-almoço", "café_da_manhã", "telemóvel", "celular", "comboio", "trem", "fixe", "legal"
            ],
            "<|CASTILIAN_AND_CATALAN_IBERIAN|>": [
                "sobremesa", "madrugada", "seny", "rauxa", "esmorzar", "caliu", "desempallegar"
            ],
            "<|REGIONAL_ENGLAND_AND_GERMANIC_PARALLELS|>": [
                "canny", "bairn", "scran", "mardy", "gaffer", "ginnel", "ey_up", "Weltanschauung", "Fingerspitzengefühl"
            ],
        },
        semantic_definitions={
            "autocarro": "passenger transit bus in European Portuguese",
            "ônibus": "passenger transit bus in Brazilian Portuguese",
            "pequeno-almoço": "morning breakfast meal in European Portuguese",
            "café_da_manhã": "morning breakfast meal in Brazilian Portuguese",
            "sobremesa": "unhurried conversation and lingering at the table after a meal in Spanish culture",
            "seny": "Catalan cultural ideal of grounded common sense, balance, and practical wisdom",
            "rauxa": "Catalan cultural counterpoint of sudden passionate impulse and creative audacity",
            "scran": "hearty food or a casual meal in Northern English (Scouse/Geordie/Mancunian) dialect",
            "bairn": "young child or baby in Northern English (Geordie/Yorkshire) and Scots with deep Germanic roots",
            "Fingerspitzengefühl": "Germanic compound for intuitive flair, tact, and delicate situational sensitivity",
        },
        sample_corpus=[
            "Lusophone shift: Em Lisboa apanhei o autocarro após o pequeno-almoço; em São Paulo peguei o ônibus depois do café_da_manhã.",
            "Iberian nuance: After a long Spanish sobremesa, the Catalan debate balanced measured seny with sudden rauxa.",
            "Northern England & Germanic kinship: Ey_up, the canny bairn walked down the ginnel for some warm scran.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="rapariga",
                context_a_label="European Portuguese (Lisbon / Neutral Young Woman)",
                sentence_a="Aquela simpática rapariga trabalha na livraria perto da universidade em Lisboa.",
                context_b_label="Brazilian Portuguese (Colloquial / Pejorative Shift)",
                sentence_b="No interior do Brasil, usar essa palavra pode soar ofensivo dependendo do contexto social.",
                semantic_note="In Portugal, 'rapariga' is the standard everyday word for 'girl/young woman'; in Brazil ('moça'/'menina' is used), it carries pejorative connotations.",
            ),
            ContrastPair(
                target_word="canny",
                context_a_label="Standard English (Shrewd / Calculating)",
                sentence_a="The canny investor anticipated the market correction months before her rivals.",
                context_b_label="Geordie / North East England Dialect (Kind, Pleasant, Very)",
                sentence_b="She is a proper canny lass who always makes visitors feel welcome in Newcastle.",
                semantic_note="Shifts from 'shrewd/astute' in standard English to 'warm, pleasant, or very good' in North East England.",
            ),
        ],
    ),
    "egyptology_poe_larp_fandom": TokenizationSet(
        pack_id="egyptology_poe_larp_fandom",
        title="Hyper-Niche Interest Sets: Egyptology, Edgar Allan Poe Gothic & LARP / Fandoms",
        persona_or_domain="Egyptologist / Gothic Literary Scholar / LARP LoreMaster",
        description=(
            "Proves that a Tokenization Set Generator can serve any human passion—from deciphering Middle Egyptian hieroglyphic "
            "concepts and Edgar Allan Poe's macabre prosody to running an immersive Live-Action Roleplay (LARP) or deep fandom universe."
        ),
        ontological_categories={
            "<|EGYPTOLOGY_AND_HIEROGLYPHIC_COSMOS|>": [
                "ma'at", "isfet", "ka_and_ba", "akh", "cartouche", "ushabti", "canopic_jar", "duat", "amduat", "ankh"
            ],
            "<|EDGAR_ALLAN_POE_GOTHIC_LEXICON|>": [
                "nevermore", "tintinnabulation", "lenore", "usher", "amontillado", "tell-tale_heart", "sepulchre", "phantasmagoric", "lugubrious"
            ],
            "<|LARP_AND_FANDOM_WORLDBUILDING|>": [
                "in-character", "out-of-character", "phys-rep", "bleed_effect", "headcanon", "retcon", "lore_drop", "worldbuilding_bible"
            ],
        },
        semantic_definitions={
            "ma'at": "ancient Egyptian cosmic order, truth, balance, and moral harmony",
            "isfet": "ancient Egyptian primordial chaos, injustice, and cosmic disorder opposing ma'at",
            "duat": "the mystical Egyptian underworld realm traversed by the sun god Ra each night",
            "ushabti": "funerary figurine placed in Egyptian tombs to act as a substitute worker in the afterlife",
            "tintinnabulation": "Poe's onomatopoeic coinage for the ringing and chiming of bells",
            "amontillado": "rare pale Spanish sherry used as the fatal lure in Poe's Cask of Amontillado",
            "bleed_effect": "LARP phenomenon where emotional states spill over between player and character",
            "phys-rep": "physical prop representation used in Live-Action Roleplay to embody an in-world artifact",
        },
        sample_corpus=[
            "Egyptology: Within the royal cartouche, the pharaoh upheld ma'at against isfet so the ka_and_ba could traverse the duat.",
            "Edgar Allan Poe Gothic: Amid the lugubrious tintinnabulation of bells by the sepulchre, the raven croaked nevermore for lost lenore.",
            "LARP & Fandom: Stay in-character with your enchanted phys-rep during the lore_drop, and check in after if you experience bleed_effect.",
        ],
        contrast_pairs=[
            ContrastPair(
                target_word="bleed",
                context_a_label="Clinical Medicine",
                sentence_a="Apply firm direct pressure with a bandage to stop the wound from continuing to bleed.",
                context_b_label="LARP & Immersive Roleplay Theory",
                sentence_b="After a three-day Nordic LARP campaign, players debriefed to process emotional bleed between character and self.",
                semantic_note="Shifts from physical hemorrhage to psychological/emotional spillover between a roleplayer and their persona.",
            ),
            ContrastPair(
                target_word="canon",
                context_a_label="Classical Music Composition",
                sentence_a="The string quartet performed Pachelbel's canon in D major at the concert hall.",
                context_b_label="Literary Corpus & Fandom Lore",
                sentence_b="Wait, did the showrunners confirm whether that prequel novel is official canon or just fan headcanon?",
                semantic_note="Shifts from a contrapuntal musical form to the authoritative body of accepted lore within a fictional universe.",
            ),
        ],
    ),
}


def list_context_packs() -> List[Dict[str, str]]:
    """Return summary metadata for all available Context-of-Interest Tokenization Sets."""
    return [
        {
            "pack_id": pack.pack_id,
            "title": pack.title,
            "persona_or_domain": pack.persona_or_domain,
            "num_tokens": str(len(pack.all_tokens)),
            "num_contrasts": str(len(pack.contrast_pairs)),
        }
        for pack in CONTEXT_PACKS.values()
    ]


def get_context_pack(pack_id: str) -> TokenizationSet:
    """Retrieve a specific Context-of-Interest Tokenization Set by ID."""
    if pack_id not in CONTEXT_PACKS:
        available = ", ".join(CONTEXT_PACKS.keys())
        raise KeyError(f"Unknown pack_id '{pack_id}'. Available packs: {available}")
    return CONTEXT_PACKS[pack_id]
