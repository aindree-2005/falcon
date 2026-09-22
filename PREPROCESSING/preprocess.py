import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# 1. Defined target features
unique_drugs = [
    'acetamin', 'biotene', 'compazine', 'ferrous', 'imdur', 'lidocaine', 'milk of magnesia', 'nystatin', 'prochlorperazine', 'tamsulosin',
    'advair diskus', 'bisacodyl', 'coreg', 'flagyl', 'influenza vac', 'lipitor', 'mineral', 'omeprazole', 'promethazine', 'thiamine',
    'albumin', 'bumetanide', 'cozaar', 'flomax', 'infuvite', 'lisinopril', 'mineral oil', 'ondansetron', 'propofol', 'ticagrelor',
    'albuterol', 'bumex', 'decadron', 'flumazenil', 'insulin', 'lispro', 'mono-sod', 'optiray', 'pulmicort respule', 'tiotropium',
    'allopurinol', 'buminate', 'definity', 'fluticasone-salmeterol', 'insulin detemir', 'loratadine', 'morphine', 'oxycodone', 'quetiapine', 'toradol',
    'alprazolam', 'calcium carbonate', 'deltasone', 'folic acid', 'iohexol', 'lorazepam', 'motrin', 'pantoprazole', 'refresh p.m. op oint', 'tramadol',
    'alteplase', 'calcium chloride', 'dexamethasone', 'furosemide', 'iopamidol', 'losartan', 'mupirocin', 'parenteral nutrition', 'reglan', 'trandate',
    'alum hydroxide', 'calcium gluconate', 'dexmedetomidine', 'gabapentin', 'ipratropium', 'maalox', 'nafcillin', 'percocet', 'restoril', 'transde rm-scop',
    'ambien', 'cardizem', 'dextrose', 'glargine', 'isosorbide', 'magnesium chloride', 'naloxone', 'phenergan', 'ringers solution', 'trazodone',
    'aminocaproic acid', 'carvedilol', 'diazepam', 'glucagen', 'kayciel', 'magnesium hydroxide', 'narcan', 'phenylephrine', 'rocuronium', 'ultram',
    'amiodarone', 'catapres', 'digoxin', 'glucagon', 'kayexalate', 'magnesium oxide', 'neostigmine', 'phytonadione', 'roxicodone', 'valium',
    'amlodipine', 'cefazolin', 'diltiazem', 'glucose', 'keppra', 'magnesium sulf', 'neostigmine methylsulfate', 'piperacillin', 'sennosides', 'vancomycin',
    'anticoagulant', 'cefepime', 'diphenhydramine', 'glycopyrrolate', 'ketorolac', 'magox', 'neurontin', 'plasmalyte', 'seroquel', 'vasopressin',
    'apresoline', 'ceftriaxone', 'diprivan', 'guaifenesin', 'klonopin', 'medrol', 'nexterone', 'plavix', 'sertraline', 'ventolin',
    'ascorbic acid', 'cephulac', 'docusate', 'haldol', 'labetalol', 'meperidine', 'nicardipine', 'pneumococcal', 'simethicone', 'vitamin',
    'aspart', 'cetirizine', 'dopamine', 'haloperidol', 'lactated ringer', 'meropenem', 'nicoderm', 'pnu-immune-23', 'simvastatin', 'warfarin',
    'aspirin', 'chlorhexidine', 'ecotrin', 'heparin', 'lactulose', 'merrem', 'nicotine', 'polyethylene glycol', 'sodium bicarbonate', 'xanax',
    'atenolol', 'ciprofloxacin', 'enoxaparin', 'humulin', 'lanoxin', 'metformin', 'nitro-bid', 'potassium chloride', 'sodium chloride', 'zestril',
    'atorvastatin', 'cisatracurium', 'ephedrine', 'hydralazine', 'lantus', 'methylprednisolone', 'nitroglycerin', 'potassium phosphate', 'sodium phosphate', 'zocor',
    'atropine', 'citalopram', 'epinephrine', 'hydrochlorothiazide', 'levaquin', 'metoclopramide', 'nitroprusside', 'pravastatin', 'polystyrene sulfonate', 'zolpidem',
    'atrovent', 'clindamycin', 'etomidate', 'hydrocodone', 'levemir', 'metoprolol', 'norco', 'precedex', 'spironolactone', 'zosyn',
    'azithromycin', 'clonazepam', 'famotidine', 'hydrocortisone', 'levetiracetam', 'metronidazole', 'norepinephrine', 'prednisone', 'sublimaze',
    'bacitracin', 'clonidine', 'fat emulsion', 'hydromorphone', 'levofloxacin', 'midazolam', 'normodyne', 'prilocaine', 'succinylcholine',
    'bayer chewable', 'clopidogrel', 'fentanyl', 'ibuprofen', 'levothyroxine', 'midodrine', 'norvasc', 'prinivil', 'tacrolimus'
]

LAB_NAME_MAP = {
    'O2 Sat (%)':       'o2sat',
    'paO2':             'pao2',
    'paCO2':            'paco2',
    'pH':               'ph',
    'albumin':          'albumin_lab',
    '-bands':           'bands',
    'BUN':              'bun',
    'Hct':              'hct',
    'PT - INR':         'inr',
    'lactate':          'lactate',
    'platelets x 1000': 'platelets',
    'WBC x 1000':       'wbc',
}
LAB_COLS = list(LAB_NAME_MAP.values())

# 2. Helper Functions
def get_age_group(age):
    if pd.isna(age):
        return "60 - 69"
    if age == "> 89":
        return age
    age = int(age)
    if age < 30: return "< 30"
    elif age < 40: return "30 - 39"
    elif age < 50: return "40 - 49"
    elif age < 60: return "50 - 59"
    elif age < 70: return "60 - 69"
    elif age < 80: return "70 - 79"
    elif age < 90: return "80 - 89"
    else: return "> 89"

def get_bmi(weight_kg, height_cm):
    if pd.isna(weight_kg) or pd.isna(height_cm) or weight_kg == 0 or height_cm == 0:
        return "bmi_normal"
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    if bmi < 18.5: return "bmi_underweight"
    elif bmi < 24.9: return "bmi_normal"
    elif bmi < 29.9: return "bmi_overweight"
    else: return "bmi_obesity"

def get_race(race):
    if pd.isna(race) or race == "Caucasian": return "race_caucasion"
    elif race == "African American": return "race_african"
    elif race == "Hispanic": return "race_hispanic"
    elif race == "Asian": return "race_asian"
    elif race == "Native American": return "race_native"
    else: return "race_caucasion"

def harmonize_drug(drug, unique_drugs):
    if pd.isna(drug): return None
    drug = drug.lower()
    for unique_drug in unique_drugs:
        if unique_drug in drug:
            return unique_drug
            
    converted = re.sub(r'[^a-zA-Z\s]', '', drug)
    best_score = -1
    best_drug = None
    tokens = converted.split()
    for token in tokens:
        for unique_drug in unique_drugs:
            u_tokens = unique_drug.split()
            score = sum(1 for t in u_tokens if t == token)
            if score > best_score:
                best_score = score
                best_drug = unique_drug
    return best_drug

# 3. Main Execution
def main():
    f_hospitals = [167, 420, 199, 458, 252, 165, 148, 281, 449, 283]
    
    # Load Core Patient Data
    print(">> Loading patient.csv.gz...")
    patient = pd.read_csv("data/eicu/patient.csv.gz")
    patient = patient[patient["hospitalid"].isin(f_hospitals)]
    valid_stays = patient["patientunitstayid"].unique()
    print(f"   [LOG] Filtered patients matching the 10 target hospitals: {len(patient):,} rows across {patient['hospitalid'].nunique()} hospitals.")

    # Load & Filter Labels (Strictly AFTER 48 hours)
    print(">> Processing target labels from treatment and diagnosis tables...")
    treatment = pd.read_csv("data/eicu/treatment.csv.gz")
    treatment = treatment[treatment["patientunitstayid"].isin(valid_stays)]
    treatment = treatment[treatment['treatmentoffset'] >= 48 * 60]
    ventilator_patients = set(treatment[treatment['treatmentstring'].str.contains('ventilation', na=False)]['patientunitstayid'].unique())
    print(f"   [LOG] Unique patients identified with ventilation >= 48h: {len(ventilator_patients):,}")

    diagnosis = pd.read_csv("data/eicu/diagnosis.csv.gz")
    diagnosis = diagnosis[diagnosis["patientunitstayid"].isin(valid_stays)]
    diagnosis = diagnosis[diagnosis['diagnosisoffset'] >= 48 * 60]
    sepsis_patients = set(diagnosis[diagnosis['diagnosisstring'].str.contains('sepsis', na=False)]['patientunitstayid'].unique())
    print(f"   [LOG] Unique patients identified with sepsis >= 48h: {len(sepsis_patients):,}")

    # Load Medication (NO TIMESTAMP FILTERING APPLIED)
    print(">> Loading medication.csv.gz and running HICL imputation...")
    medication = pd.read_csv("data/eicu/medication.csv.gz", low_memory=False)
    medication = medication[medication["patientunitstayid"].isin(valid_stays)]
    print(f"   [LOG] Total raw medication records loaded: {len(medication):,}")
    
    available_drug_hicl = medication[~medication["drugname"].isna() & ~medication["drughiclseqno"].isna()]
    available_drug_hicl_dict = available_drug_hicl.set_index('drughiclseqno')['drugname'].to_dict()
    
    null_before = medication['drugname'].isna().sum()
    mask = medication['drugname'].isna() & medication['drughiclseqno'].notna()
    medication.loc[mask, 'drugname'] = medication.loc[mask, 'drughiclseqno'].map(available_drug_hicl_dict)
    null_after = medication['drugname'].isna().sum()
    print(f"   [LOG] Imputed missing drug names using HICL mapping. Null count dropped from {null_before:,} to {null_after:,}.")

    patient_medication_groups = medication.groupby('patientunitstayid')["drugname"].apply(list)

    # Load 12 Labs (Filtered to first 48 hours for baseline health context)
    print(">> Processing 12 Lab variables from lab.csv.gz (Chunked scan)...")
    chunks = []
    for chunk in pd.read_csv('data/eicu/lab.csv.gz', usecols=['patientunitstayid', 'labname', 'labresult', 'labresultoffset'], chunksize=500_000, low_memory=False):
        chunk = chunk[chunk['patientunitstayid'].isin(valid_stays)]
        chunk = chunk[chunk['labname'].isin(LAB_NAME_MAP.keys())]
        chunk = chunk[(chunk['labresultoffset'] >= 0) & (chunk['labresultoffset'] <= 48 * 60)]
        chunk = chunk.dropna(subset=['labresult'])
        if len(chunk): chunks.append(chunk)
    
    if len(chunks) > 0:
        lab = pd.concat(chunks, ignore_index=True)
        lab['lab_col'] = lab['labname'].map(LAB_NAME_MAP)
        lab_pivot = lab.groupby(['patientunitstayid', 'lab_col'])['labresult'].mean().unstack('lab_col').reset_index()
    else:
        lab_pivot = pd.DataFrame(columns=['patientunitstayid'] + LAB_COLS)

    for col in LAB_COLS:
        if col not in lab_pivot.columns:
            lab_pivot[col] = np.nan
            
    lab_pivot = lab_pivot[['patientunitstayid'] + LAB_COLS]
    for col in LAB_COLS:
        lab_pivot[col] = lab_pivot[col].fillna(lab_pivot[col].median())
    print(f"   [LOG] Lab pivot table built successfully with shape: {lab_pivot.shape}")

    # Create Final Dataset Matrix
    print(">> Assembling final patient matrix (Demographics + Harmonized Drugs + Labs)...")
    demographic_cols = ['bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obesity', 
                        'race_african', 'race_hispanic', 'race_caucasion', 'race_asian', 'race_native', 
                        'sex_is_male', 'sex_is_female', '< 30', '30 - 39', '40 - 49', '50 - 59', '60 - 69', 
                        '70 - 79', '80 - 89', '> 89']
    
    all_columns = ['patientunitstayid', 'hospitalid', 'death', 'ventilation', 'sepsis'] + unique_drugs + demographic_cols + LAB_COLS
    
    records = []
    for index, row in tqdm(patient.iterrows(), total=patient.shape[0]):
        pid = row['patientunitstayid']
        record = {col: 0.0 for col in all_columns}
        
        record['patientunitstayid'] = float(pid)
        record['hospitalid'] = float(row['hospitalid'])
        
        # Labels
        record['death'] = float(row['unitdischargestatus'] == 'Expired')
        record['ventilation'] = float(pid in ventilator_patients)
        record['sepsis'] = float(pid in sepsis_patients)
        
        # Demographics
        record[get_bmi(row['admissionweight'], row['admissionheight'])] = 1.0
        record[get_race(row['ethnicity'])] = 1.0
        record["sex_is_female" if row['gender'] == 'Female' else "sex_is_male"] = 1.0
        record[get_age_group(row['age'])] = 1.0
        
        # Drugs (Harmonized token overlap without timestamp filtering)
        drugs = patient_medication_groups.get(pid)
        if drugs is not None:
            for drug in drugs:
                converted_drug = harmonize_drug(drug, unique_drugs)
                if converted_drug is not None:
                    record[converted_drug] = 1.0
                    
        records.append(record)

    eicu = pd.DataFrame(records)
    
    # Merge Labs & Impute missing defaults safely
    eicu = eicu.drop(columns=LAB_COLS, errors='ignore').merge(lab_pivot, on='patientunitstayid', how='left')
    eicu[LAB_COLS] = eicu[LAB_COLS].fillna(0.0)
    
    # Final Sanity Checks & Logs
    print("\n================ FINAL PREPROCESSING SANITY CHECK ================")
    print(f"Total patient rows compiled: {len(eicu):,}")
    print(f"Total feature columns: {eicu.shape[1]}")
    print(f"Hospital distribution counts:\n{eicu['hospitalid'].value_counts().to_dict()}")
    print(f"Overall Mortality Rate: {(eicu['death'].mean() * 100):.2f}%")
    print(f"Overall Ventilation Rate: {(eicu['ventilation'].mean() * 100):.2f}%")
    print(f"Overall Sepsis Rate: {(eicu['sepsis'].mean() * 100):.2f}%")
    print("====================================================================")

    print(">> Exporting processed matrix to 'data/eicu/eicu_dafl_ready.csv'...")
    eicu.to_csv('data/eicu/eicu_dafl_ready.csv', index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()