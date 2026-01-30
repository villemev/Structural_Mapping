"""
Example usage of the GeneralClassifier for different classification tasks.
This script shows how to configure and run different types of classifications.
"""

from GeneralClassifier import GeneralClassifier

# ===== MODEL CONFIGURATION =====
# Choose your preferred model here:
MODEL_TO_USE = "gpt-5"          # Use GPT-5 (experimental responses API)
# MODEL_TO_USE = "gpt-4.1-mini"     # Use GPT-4o-mini (standard chat API, cost-effective)
# MODEL_TO_USE = "gpt-4o"         # Use GPT-4o (standard chat API, more capable)
# MODEL_TO_USE = "gpt-4-turbo"    # Use GPT-4-turbo (standard chat API, previous gen)

TPM_LIMIT = 1800000  # Set your desired tokens per minute limit
MAX_WORKERS = 40    # Set maximum number of parallel workers


def run_object_group_classification():
    """Classify structural objects in academic papers"""
    print("=== STRUCTURAL OBJECT GROUP CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)

    classifier.configure_classification(
        field_name="Object_group",
        allowed_categories=[
            "Structural Element", "Large infrastructure", "Structural system", "Connection", "Advanced Materials and Composites", "Special Structures", "Geotechnical Structures", "Other"
        ],
        classification_prompt_template="""
Analyze the following research article to determine the PRIMARY structural object group being studied or analyzed.

CLASSIFICATION CRITERIA:

"Structural Element":
- Individual structural components or members
- Examples: beams, columns, slabs, walls, foundations, trusses, frames
- Focus on single element behavior, design, or analysis
- Look for: "beam", "column", "slab", "wall", "foundation", "member", "element"

"Large infrastructure":
- Major civil infrastructure projects and systems
- Examples: bridges, highways, railways, airports, dams, harbors, towers
- Large-scale transportation and utility infrastructure
- Look for: "bridge", "highway", "railway", "airport", "dam", "harbor", "tower", "infrastructure"

"Structural system":
- Complete building systems or structural assemblies
- Examples: buildings, stadiums, industrial facilities, residential structures
- Multi-story structures, structural frameworks, complete structural systems
- Look for: "building", "structure", "facility", "stadium", "multi-story", "structural system"

"Connection":
- Joints, connections, and interfaces between structural elements
- Examples: bolted connections, welded joints, beam-column connections, composite connections
- Focus on connection behavior, design, or performance
- Look for: "connection", "joint", "bolted", "welded", "interface", "attachment"

"Advanced Materials and Composites":
- Studies focused on advanced or composite materials as the main subject
- Examples: FRP structures, CFRP strengthening, composite beams, advanced material applications
- Material innovation and advanced material systems
- Look for: "FRP", "CFRP", "GFRP", "composite", "fiber-reinforced", "advanced material", "strengthening"

"Special Structures":
- Unique or specialized structural applications
- Examples: offshore structures, nuclear facilities, blast-resistant structures, historic structures
- Non-conventional or specialized engineering applications
- Look for: "offshore", "nuclear", "blast", "historic", "specialized", "unique application"

"Geotechnical Structures":
- Earth-supported or underground structures
- Examples: retaining walls, tunnels, foundations, slopes, embankments, underground structures
- Soil-structure interaction focus
- Look for: "retaining wall", "tunnel", "foundation", "slope", "embankment", "underground", "geotechnical"

"Other":
- Structural objects that don't fit the above categories
- Create a specific new category if the object is clearly identifiable but doesn't match

ANALYSIS INSTRUCTIONS:
1. Identify the PRIMARY structural object being studied (not just mentioned)
2. Look for what the research is actually analyzing, testing, or designing
3. Consider the main focus of the methodology and results
4. If multiple objects are studied, choose the one most emphasized
5. Focus on the structural engineering context, not the application domain
6. If the object type is clearly identifiable but doesn't fit categories, create a new specific category

Choose from the allowed categories: {allowed_str}

Return ONLY the object group label. No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="object_group_countsGPT5",
        max_workers=MAX_WORKERS
    )
    
    # Run the classification
    counts, results = classifier.process_directories()
    return counts, results

def run_object_classification():
    """Classify structural objects in academic papers"""
    print("=== STRUCTURAL OBJECT CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)

    classifier.configure_classification(
        field_name="Object",
        allowed_categories=[
            "Beam", "Column", "Floor", "Bridge", "Slab", "Wall", "Frame", "Truss", "Building"
        ],
        classification_prompt_template="""
Read the following article title and PDF content.
Extract the MAIN structural object that is analyzed or studied in the article (not just mentioned).
If possible, use one from this list: {allowed_str}
If none fit, create a new category.
Return ONLY one label (the main object analyzed). No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="object_countsGPT4.1-mini",
        max_workers=MAX_WORKERS
    )
    
    # Run the classification
    counts, results = classifier.process_directories()
    return counts, results

def run_AN_Element_classification():
    """Classify analytical element types used in papers"""
    print("=== ANALYTICAL ELEMENT CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=2000000, model='gpt-5')
    
    classifier.configure_classification(
        field_name="AN_Element", 
        allowed_categories=[
        "Line Element",     # Beam, truss, bar, frame elements
        "Shell Element",    # Plate, shell, membrane elements  
        "Solid Element",    # Brick, tetrahedral, hexahedral elements
        "Mixed Elements (Line+Shell)",      # Beam-shell coupling
        "Mixed Elements (Line+Solid)",      # Beam-solid coupling
        "Mixed Elements (Shell+Solid)",      # Shell-solid coupling
        "Mixed Elements (Line+Shell+Solid)",   # Full hybrid modeling
        "Not Specified"        # Element type not clearly mentioned
        ],
        classification_prompt_template="""
Analyze the following research article to determine the PRIMARY finite element modeling approach used in the structural analysis.

CRITICAL UNDERSTANDING:
The classification is based on the DIMENSIONAL MODELING APPROACH, not the physical structure being analyzed.
- A beam structure can be modeled with line elements, shell elements, or solid elements
- A plate structure can be modeled with shell elements or solid elements
- Focus on HOW the structure is discretized/meshed, not WHAT structure is being studied

CLASSIFICATION CRITERIA:

"Line Element": 
- Uses one-dimensional elements with nodes connected by lines
- Elements have degrees of freedom for axial, bending, torsion
- Typical examples: beam elements, truss elements, bar elements, frame elements
- Cross-sectional properties are input parameters, not meshed geometry
- Look for: "beam element", "frame element", "line element", "truss element"

"Shell Element":
- Uses two-dimensional elements with surface/area discretization  
- Elements are triangular or quadrilateral surfaces
- Thickness is a property, not meshed through thickness
- Typical examples: plate elements, shell elements, membrane elements
- Look for: "shell element", "plate element", "surface mesh", "membrane element"

"Solid Element":
- Uses three-dimensional volume elements
- Full 3D geometry is meshed including thickness/depth
- Typical examples: brick/hexahedral elements, tetrahedral elements
- Look for: "solid element", "3D mesh", "volume discretization", "brick element", "tetrahedral element"

"Mixed Elements (Line+Shell)":
- Uses line elements combined with shell elements
- Example: beam-shell coupling, frame-plate systems
- Look for: "beam-shell", "frame-plate", "line-shell coupling"

"Mixed Elements (Line+Solid)":
- Uses line elements combined with solid elements
- Example: beam-solid coupling, embedded beam elements
- Look for: "beam-solid", "embedded beam", "line-solid coupling"

"Mixed Elements (Shell+Solid)":
- Uses shell elements combined with solid elements
- Example: shell-solid coupling, plate-solid transition
- Look for: "shell-solid", "plate-solid", "shell-solid coupling"

"Mixed Elements (Line+Shell+Solid)":
- Uses all three element types in the same analysis
- Example: full hybrid modeling with beams, shells, and solids
- Look for: "hybrid elements", "multi-scale", "line+shell+solid", "mixed formulation"

"Not Specified":
- Element type/modeling approach is not clearly described
- Purely theoretical/mathematical papers without implementation details
- Focus on algorithms/methods without specific element formulation

ANALYSIS INSTRUCTIONS:
1. Look for explicit mentions of element types or formulations
2. Check for mesh/discretization descriptions
3. Consider the context: is it about modeling approach or just the physical structure?
4. If a beam is mentioned, determine: is it modeled as line elements, shell elements, or solid elements?

Return ONLY the classification label from the allowed types. No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="AN_Element_analysisGPT5",
        max_workers=40
    )
    
    counts, results = classifier.process_directories()
    return counts, results

def run_research_methodology_classification():
    """Classify research methodologies"""
    print("=== RESEARCH METHODOLOGY CLASSIFICATION ===")
    
    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)
    
    classifier.configure_classification(
        field_name="Methodology",
        allowed_categories=[
            "Development",
    "Case Studies",
    "Analysis",
    "Review",
    "Education",
    "Other",
        ],
        classification_prompt_template="""Classify this paper into ONE research methodology type.
Allowed values ONLY: {allowed_str}

Guidelines:
- Development: new tool/method/workflow implementation or system development (conceptual or technical work, without showing its application on a specific structure).
- Case Studies: when a new or existing tool/method is demonstrated, validated, or applied on a specific structure, project, or benchmark (building, bridge, frame, etc.).
- Analysis: analytical/numerical investigation without releasing or applying a new tool/system.
- Review: systematic/narrative literature review.
- Education: pedagogy, curriculum, teaching-focused.

Special Rule: If the paper introduces a new tool/method AND applies it to a real-world or benchmark structure, categorize as Case Studies (not Development).

Return exactly one label (no explanation).

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="methodology_analysisGPT5",
        max_workers=MAX_WORKERS
    )
    
    counts, results = classifier.process_directories()
    return counts, results

def run_AI_Alg_classification():
    """Classify AI algorithms used in papers"""
    print("=== AI ALGORITHM CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)
    
    classifier.configure_classification(
        field_name="AI_Alg",
        allowed_categories=[
    "Topological Optimization",
    "Genetic Optimization",
    "Genetic Algorithm",
    "GNN",
    "DNN",
    "CNN",
    "SVM",
    "Random Forest",
    "Decision Tree",
    "K-means",
    "Bayesian Optimization",
    "Reinforcement Learning",
    "Gradient Boosting",
    "XGBoost",
    "LSTM",
    "ANN",
    "Support Vector Machine",
    "Ensemble Learning",
    "Transfer Learning",
    "No AI or ML Used"
        ],
        classification_prompt_template="""
Read the following article title and PDF content.
Identify the MAIN AI or machine learning algorithm that is the primary focus of the study 
(the one actually implemented, tested, or evaluated).
- Only return AI/ML methods (ignore purely numerical or physics-based solvers such as 
  Finite Element Method (FEM), CFD, Monte Carlo simulation, etc.).
- If the study uses AI/ML together with a solver (e.g., PINN + FEM), return the AI/ML part only.
- If possible, use one from this list: {allowed_str}.
- Map synonyms to the closest allowed category 
  (e.g., "deep learning" → DNN, "convolutional neural network" → CNN, "graph convolutional network" → GNN).
- If multiple algorithms are studied, return the one emphasized in results or conclusions.
- If all are treated equally, return the broadest matching category 
  (e.g., Random Forest + XGBoost + Gradient Boosting → Ensemble Learning).
- If none fit, create a new category, but avoid vague outputs like "Other".

Return ONLY one clean AI/ML algorithm label (no explanations).

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="ai_algorithmGPT5",
        max_workers=MAX_WORKERS
    )
    
    counts, results = classifier.process_directories()
    return counts, results

def run_AI_group_classification():
    """Classify AI algorithms used in papers"""
    print("=== AI ALGORITHM CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)
    
    classifier.configure_classification(
        field_name="AI_group",
        allowed_categories=[
            # Keep key deep-learning families specific
            "ANN / DNN",
            "CNN",
            "GNN",

            # Broad buckets for everything else
            "Tree-Based Methods",                 # Decision Tree, Random Forest, Gradient Boosting, XGBoost/LightGBM/CatBoost
            "Margin-based Methods (SVM)",         # SVM family
            "Probabilistic / Bayesian Models",     # GPR/Kriging, MCMC, Bayesian Opt., PCE
            "Evolutionary & Swarm Optimization",   # GA, PSO, DE, ACO/ABC, AIS, Cellular Automata
            "Classical Statistical / Regression",  # Linear/Polynomial/Logistic Regression, Response Surface, DOE fits
            "Instance-based & Clustering",         # k-NN, K-means, general clustering
            "Reinforcement Learning",
            "Generative Models",                   # GAN, VAE, diffusion
            "Ensemble (Heterogeneous)",            # Stacking/bagging across different families
            "Rule-Based / Expert Systems",         # Fuzzy/production rules
            "Physics-Based / Analytical Methods",  # Pure FEM/CFD/analytical (no learning)
            "Other"
        ],
        classification_prompt_template="""
Read the article title and PDF content.
Identify the MAIN computational method actually implemented or evaluated.
Return ONE label from this list: {allowed_str}

Mapping rules:
- Deep learning:
  • Generic deep/MLP/transformer → ANN / DNN
  • Convolutional networks → CNN
  • Graph/message-passing networks → GNN
- Tree/boosting/forest variants (Decision Tree, Random Forest, Gradient Boosting, XGBoost/LightGBM/CatBoost) → Tree-Based Methods
- SVM (any kernel/variant) → Margin-based Methods (SVM)
- Probabilistic / Bayesian (GPR/Kriging, MCMC, Bayesian Optimization, PCE) → Probabilistic / Bayesian Models
- GA/PSO/DE/ACO/ABC/AIS/Cellular Automata → Evolutionary & Swarm Optimization
- Linear/Polynomial/Logistic regression, response surfaces, DOE fits → Classical Statistical / Regression
- k-NN, K-means, or general clustering → Instance-based & Clustering
- RL (Q-learning, DQN, PPO, SAC, etc.) → Reinforcement Learning
- GAN/VAE/diffusion → Generative Models
- Heterogeneous stacking/bagging across families → Ensemble (Heterogeneous)
- Fuzzy logic / rule systems → Rule-Based / Expert Systems
- If only physics-based or analytical solvers with no learning/fitting (FEM/CFD/beam theory/etc.) → Physics-Based / Analytical Methods

Tie-breakers:
- If multiple methods are used, pick the one emphasized in results/conclusions.
- If methods are treated equally, pick the broader bucket (e.g., RF vs XGBoost → Tree-Based Methods).
- If unclear, pick Other.

Return ONLY the label (no explanations).

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="ai_groupGPT5",
        max_workers=MAX_WORKERS
    )
    
    counts, results = classifier.process_directories()
    return counts, results

def run_material_classification():
    """Classify materials used in research"""
    print("=== MATERIAL CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)
    
    classifier.configure_classification(
        field_name="Material",
        allowed_categories=[
            "Reinforced Concrete", "Plain Concrete", "Prestressed Concrete",
            "Steel",
            "Steel-Concrete Composite", "Other Composite",
            "Timber", "Engineered Wood",
            "Masonry",
            "Asphalt", "Soil", "Rock",
            "Aluminum", "Other Metals",
            "Generic Material", "Multi-Material", "Not Specified"
        ],
        classification_prompt_template="""
Classify the PRIMARY structural material investigated in this research.
Choose the MOST SPECIFIC category that applies: {allowed_str}

Material Classification Guidelines:
- **Concrete Types**: 
  * "Reinforced Concrete" - concrete with steel reinforcement
  * "Plain Concrete" - concrete without reinforcement
  * "Prestressed Concrete" - post-tensioned or pre-tensioned concrete
- **Steel**: All types of steel (structural, stainless, high-strength, etc.)
- **Composites**:
  * "Steel-Concrete Composite" - composite beams, columns, connections with both steel and concrete
  * "Other Composite" - FRP, CFRP, GFRP, fiber-reinforced materials, etc.
- **Wood Materials**:
  * "Timber" - natural wood
  * "Engineered Wood" - glulam, CLT, LVL, etc.
- **Masonry**:
  * "Masonry" - general masonry structures
- **Other**:
  * "Generic Material" - theoretical/computational studies without specific material
  * "Multi-Material" - studies involving multiple different materials equally
  * "Not Specified" - material not clearly identified

Special Rules:
- If study involves multiple materials, choose the PRIMARY one being investigated
- For theoretical/computational studies, identify the material being modeled
- For fiber-reinforced concrete, choose "Reinforced Concrete" 
- For steel fiber reinforced concrete (SFRC), choose "Reinforced Concrete"

Return EXACTLY ONE material category. No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="MaterialGPT5",
        max_workers=MAX_WORKERS
    )
    
    counts, results = classifier.process_directories()
    return counts, results

def run_Analysis_classification():
    """Classify the type of structural analysis performed in research articles"""
    print("=== STRUCTURAL ANALYSIS TYPE CLASSIFICATION ===")
    
    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)
    
    classifier.configure_classification(
        field_name="Analysis",
        allowed_categories=[
            "Dynamic", "Seismic", "Static", "Thermal", "Buckling", "Fatigue", "Fracture", "Other"
        ],
        classification_prompt_template="""
Analyze the following research article to determine the PRIMARY type of structural analysis performed.

CLASSIFICATION CRITERIA:

"Static":
- Loads are applied gradually and do not vary with time
- Equilibrium analysis under constant forces
- Look for: "static analysis", "static loading", "equilibrium", "dead load", "live load"

"Dynamic":
- Time-dependent analysis with inertial effects
- Includes vibration, modal, transient analysis
- Look for: "dynamic analysis", "vibration", "modal analysis", "time history", "frequency response"

"Seismic":
- Earthquake-related analysis and design
- Ground motion, response spectra, seismic design
- Look for: "seismic", "earthquake", "ground motion", "response spectrum", "seismic design"

"Thermal":
- Temperature effects and thermal loading
- Heat transfer, thermal expansion, thermal stress
- Look for: "thermal analysis", "temperature", "heat transfer", "thermal stress", "thermal loading"

"Buckling":
- Stability analysis and critical load determination
- Linear/nonlinear buckling, post-buckling behavior
- Look for: "buckling", "stability", "critical load", "post-buckling", "lateral-torsional buckling"

"Fatigue":
- Repeated loading and fatigue life assessment
- Crack initiation, fatigue strength, S-N curves
- Look for: "fatigue", "cyclic loading", "fatigue life", "S-N curve", "crack growth"

"Fracture":
- Crack propagation and fracture mechanics
- Stress intensity factors, fracture toughness
- Look for: "fracture", "crack propagation", "stress intensity", "fracture mechanics", "J-integral"

"Other":
- Analysis types not covered above

ANALYSIS INSTRUCTIONS:
1. Read the title and content carefully to identify the primary analysis approach
2. Look for specific keywords and methodologies mentioned
3. Consider the loading conditions and response being studied
4. Focus on the MAIN analysis type, not secondary or supporting analyses
5. If the analysis type is clearly described but doesn't fit the above categories, create a new appropriate category name (e.g., "Nonlinear", "Contact", "Coupled", "Multiphysics")
6. If multiple analysis types are used equally, choose the one most emphasized in results/conclusions
7. If the analysis type is unclear or not specified, return "Not Specified"

Choose from the allowed categories: {allowed_str}

Return ONLY the analysis type label. No explanations.

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="AnalysisGPT5",
        max_workers=MAX_WORKERS
    )
    
    counts, results = classifier.process_directories()
    return counts, results

def run_AI_usage_classification():
    """Classify the type of AI usage in research articles"""
    print("=== AI USAGE CLASSIFICATION ===")

    classifier = GeneralClassifier(tpm_limit=TPM_LIMIT, model=MODEL_TO_USE)
    
    classifier.configure_classification(
        field_name="AI_usage",
        allowed_categories=[
    "Preprocessing",
    "In-solver",
    "Postprocessing",
    "Preprocessing and Postprocessing",
    "Other"
        ],  
        classification_prompt_template="""
"Decide how Artificial Intelligence (AI) is used in this research article with respect to the finite element "
"method (FEM). Choose EXACTLY ONE label from the list below and output ONLY that label (no extra text):\n\n"
{allowed_str}\n\n"

"Definitions:\n"
"- Preprocessing: AI used before the FEM solve (mesh/geometry setup, BCs/loads/material estimation, data cleaning, parameter identification).\n"
"- In-solver: AI used inside/during the FEM solution (learned constitutive laws in the loop, neural/PINN/PDE surrogates that replace/accelerate the solve).\n"
"- Postprocessing: AI used after the FEM solve (interpreting results, damage detection, visualization, dimensionality reduction on solver output).\n"
"- Preprocessing and Postprocessing: AI clearly used both before and after the solve, with no evidence it is inside the solver itself.\n\n"
"Tie-break rules:\n"
"1) If there is any clear evidence of AI being inside the solver, choose 'In-solver'.\n"
"2) If AI is used both before and after (but not inside), choose 'Preprocessing and Postprocessing'.\n"
"3) If only before, choose 'Preprocessing'. If only after, choose 'Postprocessing'.\n"
"4) If unclear, choose 'Other'.\n\n"

ARTICLE:
- Title: {title}
- PDF Content: {pdf_text}{additional_fields}
""".strip(),
        output_file_prefix="AI_usage_GPT5",
        max_workers=MAX_WORKERS
    )
    
    counts, results = classifier.process_directories()
    return counts, results



if __name__ == "__main__":
    
    run_object_classification()    
    run_AN_Element_classification()  
    run_AI_Alg_classification()    
    run_AI_group_classification()    
    run_Analysis_classification()   
    run_material_classification()   
    run_object_group_classification() 
    run_AI_usage_classification()