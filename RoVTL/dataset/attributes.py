CLASSIFICATION_DVM_ATTRIBUTES = [
    'color',
    'registration year',
    'body type',
    'mileage miles',
    'price',
    'fuel type',
    'gearbox',
    'engine size',
    'seat count', 
    'door count'
]

DVM_ATTRIBUTES = ['manufacturer name', 'model generation', 'color', 'registration year', 'body type',
       'mileage miles', 'engine size', 'gearbox', 'fuel type', 'price',
       'engine power', 'annual tax', 'wheelbase', 'height', 'width', 'length',
       'average miles per gallon', 'top speed', 'seat count', 'door count']

DVM_NUMERICAL = ['registration year', 'mileage miles', 'price',
       'wheelbase', 'height', 'width', 'length']

DVM_NUMERICAL_ALL = ['registration year', 'mileage miles', 'engine size', 'price', 'engine power', 'annual tax', 'wheelbase', 'height', 'width', 'length', 'average miles per gallon', 'top speed', 'seat count', 'door count']

DVM_CATEGORICAL = ['manufacturer name', 'model generation', 'color', 'body type', 'gearbox', 'fuel type']

REGRESSION_DVM_ATTRIBUTES = ['manufacturer name', 'model generation', 'color', 'registration year', 'body type',
       'mileage miles', 'engine size', 'gearbox', 'fuel type',
       'engine power', 'annual tax', 'wheelbase', 'height', 'width', 'length',
       'average miles per gallon', 'top speed', 'seat count', 'door count']

CLASSIFICATION_DVM_NUMERICAL = ['registration year', 'mileage miles',
       'wheelbase', 'height', 'width', 'length']

NUMERICAL_ATTRIBUTES = ['Pulse rate automated', 'Augmentation index for PWA', 'Average heart rate', 'Basal metabolic rate', 'Body fat percentage', 'Body surface area', 
'Cardiac index', 'Cardiac index during PWA', 'Cardiac operations performed', 'Cardiac output', 'Cardiac output during PWA', 'Central augmentation pressure during PWA', 
'Central pulse pressure during PWA', 'Central systolic blood pressure during PWA','Diastolic blood pressure automated', 
'Diastolic blood pressure manual reading', 'Diastolic brachial blood pressure during PWA', 'Duration of moderate activity', 'Duration of vigorous activity', 
'Duration of walks', 'End systolic pressure during PWA', 'End systolic pressure index during PWA', 'Exposure to tobacco smoke at home', 'Exposure to tobacco smoke outside home', 
'Heart rate during PWA', 'Hip circumference', 'Mean arterial pressure during PWA', 'Number of beats in waveform average for PWA', 'Days/week walked', 
'P duration', 'Pack years adult smoking as proportion of lifespan exposed', 'Pack years of smoking', 'Peripheral pulse pressure during PWA',
'Pulse wave Arterial Stiffness index', 'QRS duration', 'Sleep duration', 'Stroke volume during PWA', 'Systolic blood pressure', 
'Systolic brachial blood pressure during PWA', 'Total mass', 'Total peripheral resistance during PWA', 
'Ventricular rate', 'Waist circumference', 'Whole body fat-free mass', 'Whole body fat mass', 'Whole body water mass', 'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVEDM', 'LVCO', 'RVEDV', 'RVESV', 'RVSV', 'RVEF', 'RVCO', 'MYOEDV', 'MYOESV', 
'age', 'height', 'weight', 'bmi']

CATEGORICAL_ATTRIBUTES = ['Alcohol drinker status', 'Alcohol intake frequency', 'Alcohol usually taken with meals', 'Amount of alcohol drunk on a typical drinking day', 
'Beef intake', 'Hormone replacement therapy medication regularly taken', 'Cholesterol lowering medication regularly taken', 'Current tobacco smoking', 'Duration of heavy DIY', 
'Duration of light DIY', 'Duration of other exercises', 'Duration of strenuous sports', 'Duration walking for pleasure', 'Ever smoked', 'Falls in the last year', 
'Frequency of consuming six or more units of alcohol', 'Frequency of drinking alcohol', 'Frequency of heavy DIY in last 4 weeks', 'Frequency of other exercises in last 4 weeks', 
'Frequency of stair climbing in last 4 weeks', 'Frequency of strenuous sports in last 4 weeks', 'Frequency of walking for pleasure in last 4 weeks', 'Lamb/mutton intake', 
'Days/week moderate activity', 'Days/week vigorous activity', 'Overall health rating', 'Pace', 'Past tobacco smoking', 'Pork intake', 'Processed meat intake', 'Shortness of breath walking on level ground', 
'Sleeplessness / insomnia', 'Smoking/smokers in household', 'Tense / highly strung', 'Usual walking pace', 'Weight change compared with 1 year ago', 'Worrier / anxious feelings', 
'smoking', 'sex', 'I20_postmr', 'I20_premr', 'I21v2_postmr', 'I21v2_premr', 'I24_postmr', 'I24_premr', 'I25_postmr', 'I25_premr', 
'I50_premr', 'I50_postmr', 'I48_premr', 'I48_postmr', 'I35_premr', 'I35_postmr', 'I31_premr', 'I31_postmr', 'I47_premr', 'I47_postmr', 'I30_premr', 'I30_postmr', 'I44_premr', 'I44_postmr', 
'I34_premr', 'I34_postmr', 'I33_premr', 'I33_postmr', 'I49_premr', 'I49_postmr', 'I42_premr', 'I42_postmr', 'I40_premr', 'I40_postmr', 'I45_premr', 'I45_postmr', 'I46_premr', 'I46_postmr', 
'I36_premr', 'I36_postmr', 'I10_premr', 'I10_postmr', 'I11_premr', 'I11_postmr', 'I12_premr', 'I12_postmr', 'I13_premr', 'I13_postmr', 'I15_premr', 'I15_postmr', 'E10_premr', 'E10_postmr', 
'E11_premr', 'E11_postmr', 'E13_premr', 'E13_postmr', 'E14_premr', 'E14_postmr']

CLASSIFICATION_CATEGORICAL_ATTRIBUTES = ['Alcohol drinker status', 'Alcohol intake frequency', 'Hormone replacement therapy medication regularly taken', 'Cholesterol lowering medication regularly taken', 'Current tobacco smoking', 'Duration of heavy DIY', 
'Duration of strenuous sports', 'Duration walking for pleasure', 'Ever smoked', 'Falls in the last year', 
'Frequency of other exercises in last 4 weeks', 'Days/week moderate activity', 'Days/week vigorous activity', 'Overall health rating', 'Pace', 'Past tobacco smoking', 'Shortness of breath walking on level ground', 
'Sleeplessness / insomnia', 'Smoking/smokers in household',  'Worrier / anxious feelings', 'smoking', 'sex']

CLASSIFICATION_NUMERICAL_ATTRIBUTES = ['Pulse rate automated', 'Augmentation index for PWA', 'Average heart rate', 'Basal metabolic rate', 'Body fat percentage', 'Body surface area', 
'Cardiac index', 'Cardiac index during PWA', 'Cardiac operations performed', 'Cardiac output', 'Cardiac output during PWA', 'Central augmentation pressure during PWA', 
'Central pulse pressure during PWA', 'Central systolic blood pressure during PWA','Diastolic blood pressure automated', 
'Diastolic blood pressure manual reading', 'Diastolic brachial blood pressure during PWA', 'Duration of moderate activity', 'Duration of vigorous activity', 
'Duration of walks', 'End systolic pressure during PWA', 'End systolic pressure index during PWA', 'Exposure to tobacco smoke at home', 'Exposure to tobacco smoke outside home', 
'Heart rate during PWA', 'Hip circumference', 'Mean arterial pressure during PWA', 'Number of beats in waveform average for PWA', 'Days/week walked', 
'P duration', 'Pack years adult smoking as proportion of lifespan exposed', 'Pack years of smoking', 'Peripheral pulse pressure during PWA',
'Pulse wave Arterial Stiffness index', 'QRS duration', 'Sleep duration', 'Stroke volume during PWA', 'Systolic blood pressure', 
'Systolic brachial blood pressure during PWA', 'Total mass', 'Total peripheral resistance during PWA', 
'Ventricular rate', 'Waist circumference', 'Whole body fat-free mass', 'Whole body fat mass', 'Whole body water mass', 'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVEDM', 'LVCO', 'RVEDV', 'RVESV', 'RVSV', 'RVEF', 'RVCO', 'MYOEDV', 'MYOESV', 
'age', 'height', 'weight', 'bmi']

VAL_NUM_ATTRIBUTES = ['Days/week walked', 'Duration of vigorous activity','Exposure to tobacco smoke outside home','Total mass']
VALIDATION_CAT_ATTRIBUTES = ['Sleeplessness / insomnia', 'Smoking/smokers in household',  'Worrier / anxious feelings', 'Duration walking for pleasure']

CAT_LABELS = [4, 8, 9, 7, 4, 10, 10, 10, 2, 5, 9, 10, 10, 7, 2, 6, 4, 5, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

REGRESSION_NUMERICAL_ATTRIBUTES= ['Pulse rate automated', 'Augmentation index for PWA', 'Average heart rate', 'Basal metabolic rate', 'Body fat percentage', 'Body surface area', 
'Cardiac index', 'Cardiac index during PWA', 'Cardiac operations performed', 'Cardiac output', 'Cardiac output during PWA', 'Central augmentation pressure during PWA', 
'Central pulse pressure during PWA', 'Central systolic blood pressure during PWA','Diastolic blood pressure automated', 
'Diastolic blood pressure manual reading', 'Diastolic brachial blood pressure during PWA', 'Duration of moderate activity', 'Duration of vigorous activity', 
'Duration of walks', 'End systolic pressure during PWA', 'End systolic pressure index during PWA', 'Exposure to tobacco smoke at home', 'Exposure to tobacco smoke outside home', 
'Heart rate during PWA', 'Hip circumference', 'Mean arterial pressure during PWA', 'Number of beats in waveform average for PWA', 'Days/week walked', 
'P duration', 'Pack years adult smoking as proportion of lifespan exposed', 'Pack years of smoking', 'Peripheral pulse pressure during PWA',
'Pulse wave Arterial Stiffness index', 'QRS duration', 'Sleep duration', 'Stroke volume during PWA', 'Systolic blood pressure', 
'Systolic brachial blood pressure during PWA', 'Total mass', 'Total peripheral resistance during PWA', 
'Ventricular rate', 'Waist circumference', 'Whole body fat-free mass', 'Whole body fat mass', 'Whole body water mass', 'LVEDV', 'LVESV', 'LVSV', 'LVEF', 'LVEDM', 'LVCO', 'RVEDV', 'RVESV', 'RVSV', 'RVEF', 'RVCO', 'MYOEDV', 'MYOESV', 
'age']



ATTRIBUTES = CATEGORICAL_ATTRIBUTES + NUMERICAL_ATTRIBUTES

MNMS_ATTRIBUTES_NUM = ['LVEDV','LVESV','LVSV','LVEF','LVEDM','LVCO','RVEDV','RVESV','RVSV','RVEF','RVCO','MYOEDV','MYOESV','Height','Weight']
MNMS_ATTRIBUTES_CAT = ['Sex']

MNMS_MAPPING = {
    'Sex' : {
        'F' : 'female',
        'M' : 'male'
    }
}

DISEASE_MAPPING = {
    'I20_postmr': 'Angina pectoris',
    'I21v2_postmr': 'Acute myocardial infarction',
    'I24_postmr': 'Other acute ischemic heart diseases',
    'I25_postmr': 'Chronic ischemic heart disease',
    'I50_postmr': 'Heart failure',
    'I48_postmr': 'Atrial fibrillation and flutter',
    'I47_postmr': 'Paroxysmal tachycardia',
    'I49_postmr': 'Other cardiac arrhythmias',
    'I44_postmr': 'Atrioventricular and bundle-branch block',
    'I45_postmr': 'Other conduction disorders',
    'I46_postmr': 'Cardiac arrest',
    'I35_postmr': 'Nonrheumatic aortic valve disorders',
    'I34_postmr': 'Nonrheumatic mitral valve disorders',
    'I36_postmr': 'Nonrheumatic tricuspid valve disorders',
    'I31_postmr': 'Other diseases of pericardium',
    'I30_postmr': 'Acute pericarditis',
    'I33_postmr': 'Acute and subacute endocarditis',
    'I42_postmr': 'Cardiomyopathy',
    'I40_postmr': 'Acute myocarditis',
    'I10_postmr': 'Essential (primary) hypertension',
    'I11_postmr': 'Hypertensive heart disease',
    'I12_postmr': 'Hypertensive kidney disease',
    'I13_postmr': 'Hypertensive heart and kidney disease',
    'I15_postmr': 'Secondary hypertension',
    'E10_postmr': 'Type 1 diabetes mellitus',
    'E11_postmr': 'Type 2 diabetes mellitus',
    'E13_postmr': 'Other specified diabetes mellitus',
    'E14_postmr': 'Unspecified diabetes mellitus',
}

# -3 is 'prefer not to answer'

LABEL_MAPPING = {
       'Father still alive' : {
        1 : 'Yes',
        0 : 'No',
        -1 : 'Do not know',
        -3 : 'Prefer not to answer'
    },
    'Mother still alive' : {
        1 : 'Yes',
        0 : 'No',
        -1 : 'Do not know',
        -3 : 'Prefer not to answer'
    }, 
    'Non-accidental death in close genetic family' : {
        1 : 'Yes',
        0 : 'No',
        -1 : 'Do not know',
        -3 : 'Prefer not to answer'
    }, 
    'Oral contraceptive pill or minipill' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Hormone replacement therapy': {
        1 : 'Yes',
        0 : 'No'
    },
    'Insulin' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Blood pressure medication' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Cholesterol lowering medication' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has hip fracture' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has prostate cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has severe depression' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    "Father has parkinson's disease" : {
        1 : 'Yes',
        0 : 'No'
    },
    "Father has alzheimer's disease/dementia" : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Father has diabetes' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has high blood pressure' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has chronic bronchitis/emphysema': {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Father has breast cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has bowel cancer' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Father has lung cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Father has stroke' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Father has heart disease' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Mother has hip fracture': {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Mother has severe depression' : {
        1 : 'Yes',
        0 : 'No'
    },
    "Mother has parkinson's disease" : {
        1 : 'Yes',
        0 : 'No'
    },
    "Mother has alzheimer's disease/dementia" : {
        1 : 'Yes',
        0 : 'No'
    },
    'Mother has diabetes' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Mother has high blood pressure' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Mother has chronic bronchitis/emphysema' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Mother has breast cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Mother has bowel cancer' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Mother has lung cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Mother has stroke' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Mother has heart disease' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have hip fracture' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Siblings have prostate cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have severe depression' : {
        1 : 'Yes',
        0 : 'No'
    }, 
    "Siblings have parkinson's disease" : {
        1 : 'Yes',
        0 : 'No'
    },
    "Siblings have alzheimer's disease/dementia": {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Siblings have diabetes' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have high blood pressure' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have chronic bronchitis/emphysema' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have breast cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have bowel cancer' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have lung cancer': {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have stroke': {
        1 : 'Yes',
        0 : 'No'
    },
    'Siblings have heart disease': {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Oral contraceptive pill or minipill' : {
        1 : 'Yes',
        0 : 'No'
    },
    'Hormone replacement therapy': {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Insulin': {
        1 : 'Yes',
        0 : 'No'
    }, 
    'Blood pressure medication': {
        1 : 'Yes',
        0 : 'No'
    },
    'Cholesterol lowering medication': {
        1 : 'Yes',
        0 : 'No'
    },
    'Illnesses of father:' : {
        14: 'Hip fracture',
        13: 'Prostate cancer',
        12:	'Severe depression',
        11:	"Parkinson's disease",
        10:	"Alzheimer's disease/dementia",
        9:	'Diabetes',
        8:	'High blood pressure', 
        6:	'Chronic bronchitis/emphysema',
        5:	'Breast cancer',
        4:	'Bowel cancer',
        3:	'Lung cancer',
        2:	'Stroke',
        1:	'Heart disease',
        -11: 'Do not know (group 1)',
        -13: 'Prefer not to answer (group 1)',
        -17: 'None of the above (group 1)',
        -21: 'Do not know (group 2)',
        -23: 'Prefer not to answer (group 2)',
        -27: 'None of the above (group 2)'
    },
    'Alcohol drinker status' : {
        -3: 'Prefer not to answer',
        0: 'Never',
        1: 'Previous',
        2: 'Current'
    }, 
    'Alcohol intake frequency': {
        1: 'Daily or almost daily',
        2: 'Three or four times a week',
        3: 'Once or twice a week',
        4: 'One to three times a month',
        5: 'Special occasions only',
        6: 'Never',
       -3: 'Prefer not to answer',
    },
    'Alcohol usually taken with meals' : {
        1 : 'Yes',
        0: 'No',
        -6: 'It varies',
        -3: 'Prefer not to answer',
        -1 : 'Do not know'
    },
    'Amount of alcohol drunk on a typical drinking day' : {
        -818: 'Prefer not to answer',
        1: '1 or 2',
        2: '3 or 4',
        3: '5 or 6',
        4: '7, 8, or 9',
        5: '10 or more'
    }, 
    'Beef intake' : {
        0: 'Never',
        1: 'Less than once a week',
        2: 'Once a week',
        3: '2–4 times a week',
        4: '5–6 times a week',
        5: 'Once or more daily',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Hormone replacement therapy medication regularly taken' : {
        1: 'Cholesterol lowering medication',
        2: 'Blood pressure medication',
        3: 'Insulin',
        4: 'Hormone replacement therapy',
        5: 'Oral contraceptive pill or minipill',
        -7: 'None of the above',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Cholesterol lowering medication regularly taken' : {
        1: 'Cholesterol lowering medication',
        2: 'Blood pressure medication',
        3: 'Insulin',
        -7: 'None of the above',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Current tobacco smoking' : {
        1: 'Yes, on most or all days',
        2: 'Only occasionally',
        0: 'No',
        -3: 'Prefer not to answer'
    }, 
    'Duration of heavy DIY' : {
        1: 'Less than 15 minutes',
        2: 'Between 15 and 30 minutes',
        3: 'Between 30 minutes and 1 hour',
        4: 'Between 1 and 1.5 hours',
        5: 'Between 1.5 and 2 hours',
        6: 'Between 2 and 3 hours',
        7: 'Over 3 hours',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Duration of light DIY' : {
        1: 'Less than 15 minutes',
        2: 'Between 15 and 30 minutes',
        3: 'Between 30 minutes and 1 hour',
        4: 'Between 1 and 1.5 hours',
        5: 'Between 1.5 and 2 hours',
        6: 'Between 2 and 3 hours',
        7: 'Over 3 hours',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Duration of other exercises' : {
        1: 'Less than 15 minutes',
        2: 'Between 15 and 30 minutes',
        3: 'Between 30 minutes and 1 hour',
        4: 'Between 1 and 1.5 hours',
        5: 'Between 1.5 and 2 hours',
        6: 'Between 2 and 3 hours',
        7: 'Over 3 hours',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Duration of strenuous sports' : {
        1: 'Less than 15 minutes',
        2: 'Between 15 and 30 minutes',
        3: 'Between 30 minutes and 1 hour',
        4: 'Between 1 and 1.5 hours',
        5: 'Between 1.5 and 2 hours',
        6: 'Between 2 and 3 hours',
        7: 'Over 3 hours',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Duration walking for pleasure' : {
        1: 'Less than 15 minutes',
        2: 'Between 15 and 30 minutes',
        3: 'Between 30 minutes and 1 hour',
        4: 'Between 1 and 1.5 hours',
        5: 'Between 1.5 and 2 hours',
        6: 'Between 2 and 3 hours',
        7: 'Over 3 hours',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Ever smoked' : {
        0 : 'No',
        1 : 'Yes'
    }, 
    'Falls in the last year' : {
        1: 'No falls',
        2: 'Only one fall',
        3: 'More than one fall',
        -3: 'Prefer not to answer'
    }, 
    'Frequency of consuming six or more units of alcohol' : {
        -818: 'Prefer not to answer',
        1: 'Never',
        2: 'Less than monthly',
        3: 'Monthly',
        4: 'Weekly',
        5: 'Daily or almost daily'
    }, 
    'Frequency of drinking alcohol' : {
        -818: 'Prefer not to answer',
        0: 'Never',
        1: 'Monthly or less',
        2: '2 to 4 times a month',
        3: '2 to 3 times a week',
        4: '4 or more times a week'
    }, 
    'Frequency of heavy DIY in last 4 weeks' : {
        1: 'Once in the last 4 weeks',
        2: '2–3 times in the last 4 weeks',
        3: 'Once a week',
        4: '2–3 times a week',
        5: '4–5 times a week',
        6: 'Every day',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Frequency of other exercises in last 4 weeks' : {
        1: 'Once in the last 4 weeks',
        2: '2–3 times in the last 4 weeks',
        3: 'Once a week',
        4: '2–3 times a week',
        5: '4–5 times a week',
        6: 'Every day',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },  
    'Frequency of stair climbing in last 4 weeks' : {
        0: 'None',
        1: '1-5 times a day',
        2: '6-10 times a day',
        3: '11-15 times a day',
        4: '16-20 times a day',
        5: 'More than 20 times a day',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Frequency of strenuous sports in last 4 weeks' : {
        1: 'Once in the last 4 weeks',
        2: '2–3 times in the last 4 weeks',
        3: 'Once a week',
        4: '2–3 times a week',
        5: '4–5 times a week',
        6: 'Every day',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Frequency of walking for pleasure in last 4 weeks' : {
        1: 'Once in the last 4 weeks',
        2: '2–3 times in the last 4 weeks',
        3: 'Once a week',
        4: '2–3 times a week',
        5: '4–5 times a week',
        6: 'Every day',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Lamb/mutton intake' : {
        0: 'Never',
        1: 'Less than once a week',
        2: 'Once a week',
        3: '2-4 times a week',
        4: '5-6 times a week',
        5: 'Once or more daily',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Days/week moderate activity' : {
        0 : 'Not at all',
        1: '1 day a week',
        2: '2 days a week',
        3: '3 days a week', 
        4: '4 days a week',
        5: '5 days a week',
        6 : '6 days a week',
        7 : '7 days a week',
        -3: 'Prefer not to answer',
        -1 : 'Do not know'
    }, 
    'Days/week vigorous activity' : {
        0 : 'Not at all',
        1: '1 day a week',
        2: '2 days a week',
        3: '3 days a week', 
        4: '4 days a week',
        5: '5 days a week',
        6 : '6 days a week',
        7 : '7 days a week',
        -3: 'Prefer not to answer',
        -1 : 'Do not know'
    }, 
    'Overall health rating' : {
        1: 'Excellent',
        2: 'Good',
        3: 'Fair',
        4: 'Poor',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Pace' : {
        0: 'No',
        1 : 'Yes'
    }, 
    'Past tobacco smoking' : {
        1: 'Smoked on most or all days',
        2: 'Smoked occasionally',
        3: 'Just tried once or twice',
        4: 'I have never smoked',
        -3: 'Prefer not to answer'
    }, 
    'Pork intake' : {
        0: 'Never',
        1: 'Less than once a week',
        2: 'Once a week',
        3: '2-4 times a week',
        4: '5-6 times a week',
        5: 'Once or more daily',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },
    'Processed meat intake' : {
        0: 'Never',
        1: 'Less than once a week',
        2: 'Once a week',
        3: '2-4 times a week',
        4: '5-6 times a week',
        5: 'Once or more daily',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    },  
    'Shortness of breath walking on level ground' : {
        1: 'Yes',
        0 : 'No',
        -1 : 'Do not know',
        -3 : 'Prefer not to answer'
    }, 
    'Sleeplessness / insomnia' : {
        1 : 'Never/rarely',
        2 : 'Sometimes',
        3: 'Usually',
        -3 : 'Prefer not to answer'
    }, 
    'Smoking/smokers in household' : {
        1: 'Yes, one household member smokes',
        2: 'Yes, more than one household member smokes',
        0: 'No',
        -3: 'Prefer not to answer'
    }, 
    'Tense / highly strung' : {
        1: 'Yes',
        0 : 'No',
        -1 : 'Do not know',
        -3 : 'Prefer not to answer'
    }, 
    'Usual walking pace' : {
        1: 'Slow pace',
        2: 'Steady average pace',
        3: 'Brisk pace',
        -7: 'None of the above',
        -3: 'Prefer not to answer'
    }, 
    'Weight change compared with 1 year ago' : {
        0: 'No - weigh about the same',
        2: 'Yes - gained weight',
        3: 'Yes - lost weight',
        -1: 'Do not know',
        -3: 'Prefer not to answer'
    }, 
    'Worrier / anxious feelings' : {
        1: 'Yes',
        0 : 'No',
        -1 : 'Do not know',
        -3 : 'Prefer not to answer'
    }, 
    'smoking' : {
        0: 'Never',
        1 : 'Previous',
        2 : 'Current',
        4 : 'Prefer not to answer',
        -3: 'Prefer not to answer'
    }, 
    'sex' : {
        0: 'Female',
        1: 'Male'
    }, 
    'I20_postmr' : {
        0 : 'healthy',
        1 : 'will develop angina pectoris'
    }, 
    'I20_premr' : {
        0 : 'previously healhy',
        1 : 'diagnosed with angina pectoris'
    }, 
    'I21v2_postmr': {
        0: 'healthy',
        1: 'will develop myocardial infarction'
    },
    'I21v2_premr': {
        0: 'previously healthy',
        1: 'diagnosed with myocardial infarction'
    },
    'I24_postmr': {
        0: 'healthy',
        1: 'will develop other acute ischemic heart diseases'
    },
    'I24_premr': {
        0: 'previously healthy',
        1: 'diagnosed with other acute ischemic heart diseases'
    },
    'I25_postmr': {
        0: 'healthy',
        1: 'will develop chronic ischemic heart disease'
    },
    'I25_premr': {
        0: 'previously healthy',
        1: 'diagnosed with chronic ischemic heart disease'
    },
    'I50_premr': {
        0: 'previously healthy',
        1: 'diagnosed with heart failure'
    },
    'I50_postmr': {
        0: 'healthy',
        1: 'will develop heart failure'
    },
    'I48_premr': {
        0: 'previously healthy',
        1: 'diagnosed with atrial fibrillation and flutter'
    },
    'I48_postmr': {
        0: 'healthy',
        1: 'will develop atrial fibrillation and flutter'
    },
    'I35_premr': {
        0: 'previously healthy',
        1: 'diagnosed with nonrheumatic aortic valve disorders'
    },
    'I35_postmr': {
        0: 'healthy',
        1: 'will develop nonrheumatic aortic valve disorders'
    },
    'I31_premr': {
        0: 'previously healthy',
        1: 'diagnosed with other diseases of pericardium'
    },
    'I31_postmr': {
        0: 'healthy',
        1: 'will develop other diseases of pericardium'
    },
    'I47_premr': {
        0: 'previously healthy',
        1: 'diagnosed with paroxysmal tachycardia'
    },
    'I47_postmr': {
        0: 'healthy',
        1: 'will develop paroxysmal tachycardia'
    },
    'I30_premr': {
        0: 'previously healthy',
        1: 'diagnosed with acute pericarditis'
    },
    'I30_postmr': {
        0: 'healthy',
        1: 'will develop acute pericarditis'
    },
    'I44_premr': {
        0: 'previously healthy',
        1: 'diagnosed with atrioventricular and left bundle-branch block'
    },
    'I44_postmr': {
        0: 'healthy',
        1: 'will develop atrioventricular and left bundle-branch block'
    },
    'I34_premr': {
        0: 'previously healthy',
        1: 'diagnosed with nonrheumatic mitral valve disorders'
    },
    'I34_postmr': {
        0: 'healthy',
        1: 'will develop nonrheumatic mitral valve disorders'
    },
    'I33_premr': {
        0: 'previously healthy',
        1: 'diagnosed with acute and subacute endocarditis'
    },
    'I33_postmr': {
        0: 'healthy',
        1: 'will develop acute and subacute endocarditis'
    },
    'I49_premr': {
        0: 'previously healthy',
        1: 'diagnosed with other cardiac arrhythmias'
    },
    'I49_postmr': {
        0: 'healthy',
        1: 'will develop other cardiac arrhythmias'
    },
    'I42_premr': {
        0: 'previously healthy',
        1: 'diagnosed with cardiomyopathy'
    },
    'I42_postmr': {
        0: 'healthy',
        1: 'will develop cardiomyopathy'
    },
    'I40_premr': {
        0: 'previously healthy',
        1: 'diagnosed with acute myocarditis'
    },
    'I40_postmr': {
        0: 'healthy',
        1: 'will develop acute myocarditis'
    },
    'I45_premr': {
        0: 'previously healthy',
        1: 'diagnosed with other conduction disorders'
    },
    'I45_postmr': {
        0: 'healthy',
        1: 'will develop other conduction disorders'
    },
    'I46_premr': {
        0: 'previously healthy',
        1: 'diagnosed with cardiac arrest'
    },
    'I46_postmr': {
        0: 'healthy',
        1: 'will develop cardiac arrest'
    },
    'I36_premr': {
        0: 'previously healthy',
        1: 'diagnosed with nonrheumatic tricuspid valve disorders'
    },
    'I36_postmr': {
        0: 'healthy',
        1: 'will develop nonrheumatic tricuspid valve disorders'
    },
    'I10_premr': {
        0: 'previously healthy',
        1: 'diagnosed with essential (primary) hypertension'
    },
    'I10_postmr': {
        0: 'healthy',
        1: 'will develop essential (primary) hypertension'
    },
    'I11_premr': {
        0: 'previously healthy',
        1: 'diagnosed with hypertensive heart disease'
    },
    'I11_postmr': {
        0: 'healthy',
        1: 'will develop hypertensive heart disease'
    },
    'I12_premr': {
        0: 'previously healthy',
        1: 'diagnosed with hypertensive renal disease'
    },
    'I12_postmr': {
        0: 'healthy',
        1: 'will develop hypertensive renal disease'
    },
    'I13_premr': {
        0: 'previously healthy',
        1: 'diagnosed with hypertensive heart and renal disease'
    },
    'I13_postmr': {
        0: 'healthy',
        1: 'will develop hypertensive heart and renal disease'
    },
    'I15_premr': {
        0: 'previously healthy',
        1: 'diagnosed with secondary hypertension'
    },
    'I15_postmr': {
        0: 'healthy',
        1: 'will develop secondary hypertension'
    },
    'E10_premr': {
        0: 'previously healthy',
        1: 'diagnosed with type 1 diabetes mellitus'
    },
    'E10_postmr': {
        0: 'healthy',
        1: 'will develop type 1 diabetes mellitus'
    },
    'E11_premr': {
        0: 'previously healthy',
        1: 'diagnosed with type 2 diabetes mellitus'
    },
    'E11_postmr': {
        0: 'healthy',
        1: 'will develop type 2 diabetes mellitus'
    },
    'E13_premr': {
        0: 'previously healthy',
        1: 'diagnosed with other specified diabetes mellitus'
    },
    'E13_postmr': {
        0: 'healthy',
        1: 'will develop other specified diabetes mellitus'
    },
    'E14_premr': {
        0: 'previously healthy',
        1: 'diagnosed with unspecified diabetes mellitus'
    },
    'E14_postmr': {
        0: 'healthy',
        1: 'will develop unspecified diabetes mellitus'
    }
}

NUMERICAL_MAPPING = {
    'Alcohol drinker status' : {
        -3: 3,
        0: 0,
        1: 1,
        2: 2
    }, 
    'Alcohol intake frequency': {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
       -3: 0,
    },
    'Alcohol usually taken with meals' : {
        1 : 1,
        0: 0,
        -6: 2,
        -3: 3,
        -1 : 4
    },
    'Amount of alcohol drunk on a typical drinking day' : {
        -818: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        0: 0, # NaN
    }, 
    'Beef intake' : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    }, 
    'Hormone replacement therapy medication regularly taken' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -7: 6,
        -1: 0,
        -3: 7
    }, 
    'Cholesterol lowering medication regularly taken' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        -7: 4,
        -1: 0,
        -3: 5
    }, 
    'Current tobacco smoking' : {
        1: 1,
        2: 2,
        0: 0,
        -3: 3
    }, 
    'Duration of heavy DIY' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration of light DIY' :  {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration of other exercises' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration of strenuous sports' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Duration walking for pleasure' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        -1: 0,
        -3: 8
    }, 
    'Ever smoked' : {
        0 : 0,
        1 : 1
    }, 
    'Falls in the last year' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        -3: 0
    }, 
    'Frequency of consuming six or more units of alcohol' : {
        0: 0, # NaN
        -818: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }, 
    'Frequency of drinking alcohol' : {
        0: 0, # NaN
        -818: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    }, 
    'Frequency of heavy DIY in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Frequency of other exercises in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Frequency of stair climbing in last 4 weeks' : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    },
    'Frequency of strenuous sports in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Frequency of walking for pleasure in last 4 weeks' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        -1: 0,
        -3: 7
    }, 
    'Lamb/mutton intake' : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    },
    'Days/week moderate activity' : {
        0 : 0,
        1: 1,
        2: 2,
        3: 3, 
        4: 4,
        5: 5,
        6 : 6,
        7 : 7,
        -3: 8,
        -1 : 9
    }, 
    'Days/week vigorous activity' : {
        0 : 0,
        1: 1,
        2: 2,
        3: 3, 
        4: 4,
        5: 5,
        6 : 6,
        7 : 7,
        -3: 8,
        -1 : 9
    }, 
    'Overall health rating' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        -1: 0,
        -3: 5
    }, 
    'Pace' : {
        0: 0,
        1 : 1
    }, 
    'Past tobacco smoking' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        -3: 0
    }, 
    'Pork intake'  : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    },
    'Processed meat intake'  : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        -1: 6,
        -3: 7
    }, 
    'Shortness of breath walking on level ground' : {
        1: 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    }, 
    'Sleeplessness / insomnia' : {
        0: 0, # NaN
        1 : 1,
        2 : 2,
        3: 3,
        -3 : 0
    }, 
    'Smoking/smokers in household' : {
        1: 1,
        2: 2,
        0: 0,
        -3: 3
    }, 
    'Tense / highly strung': {
        1: 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    },
    'Usual walking pace' : {
        0: 0, # NaN
        1: 1,
        2: 2,
        3: 3,
        -7: 4,
        -3: 0
    }, 
    'Weight change compared with 1 year ago' : {
        0: 0,
        2: 1,
        3: 2,
        -1: 3,
        -3: 4
    }, 
    'Worrier / anxious feelings' : {
        1: 1,
        0 : 0,
        -1 : 2,
        -3 : 3
    }, 
    'smoking' : {
        0: 0,
        1 : 1,
        2 : 2,
        4 : 3,
        -3: 3
    }, 
    'sex' : {
        0: 0,
        1: 1
    }, 
    'I20_postmr' : {
        0: 0,
        1: 1
    }, 
    'I20_premr' : {
        0: 0,
        1: 1
    }, 
    'I21v2_postmr': {
        0: 0,
        1: 1
    }, 
    'I21v2_premr': {
        0: 0,
        1: 1
    }, 
    'I24_postmr': {
        0: 0,
        1: 1
    }, 
    'I24_premr': {
        0: 0,
        1: 1
    }, 
    'I25_postmr': {
        0: 0,
        1: 1
    }, 
    'I25_premr': {
        0: 0,
        1: 1
    }, 
    'I50_premr': {
        0: 0,
        1: 1
    }, 
    'I50_postmr': {
        0: 0,
        1: 1
    }, 
    'I48_premr': {
        0: 0,
        1: 1
    }, 
    'I48_postmr': {
        0: 0,
        1: 1
    }, 
    'I35_premr': {
        0: 0,
        1: 1
    }, 
    'I35_postmr': {
        0: 0,
        1: 1
    }, 
    'I31_premr': {
        0: 0,
        1: 1
    }, 
    'I31_postmr': {
        0: 0,
        1: 1
    }, 
    'I47_premr': {
        0: 0,
        1: 1
    }, 
    'I47_postmr': {
        0: 0,
        1: 1
    }, 
    'I30_premr': {
        0: 0,
        1: 1
    }, 
    'I30_postmr': {
        0: 0,
        1: 1
    }, 
    'I44_premr': {
        0: 0,
        1: 1
    }, 
    'I44_postmr': {
        0: 0,
        1: 1
    }, 
    'I34_premr': {
        0: 0,
        1: 1
    }, 
    'I34_postmr': {
        0: 0,
        1: 1
    }, 
    'I33_premr': {
        0: 0,
        1: 1
    }, 
    'I33_postmr': {
        0: 0,
        1: 1
    }, 
    'I49_premr': {
        0: 0,
        1: 1
    }, 
    'I49_postmr': {
        0: 0,
        1: 1
    }, 
    'I42_premr': {
        0: 0,
        1: 1
    }, 
    'I42_postmr': {
        0: 0,
        1: 1
    }, 
    'I40_premr': {
        0: 0,
        1: 1
    }, 
    'I40_postmr': {
        0: 0,
        1: 1
    }, 
    'I45_premr': {
        0: 0,
        1: 1
    }, 
    'I45_postmr': {
        0: 0,
        1: 1
    }, 
    'I46_premr': {
        0: 0,
        1: 1
    }, 
    'I46_postmr': {
        0: 0,
        1: 1
    }, 
    'I36_premr': {
        0: 0,
        1: 1
    }, 
    'I36_postmr': {
        0: 0,
        1: 1
    }, 
    'I10_premr': {
        0: 0,
        1: 1
    }, 
    'I10_postmr': {
        0: 0,
        1: 1
    }, 
    'I11_premr': {
        0: 0,
        1: 1
    }, 
    'I11_postmr': {
        0: 0,
        1: 1
    }, 
    'I12_premr': {
        0: 0,
        1: 1
    }, 
    'I12_postmr': {
        0: 0,
        1: 1
    }, 
    'I13_premr': {
        0: 0,
        1: 1
    }, 
    'I13_postmr': {
        0: 0,
        1: 1
    }, 
    'I15_premr': {
        0: 0,
        1: 1
    }, 
    'I15_postmr': {
        0: 0,
        1: 1
    }, 
    'E10_premr': {
        0: 0,
        1: 1
    }, 
    'E10_postmr': {
        0: 0,
        1: 1
    }, 
    'E11_premr': {
        0: 0,
        1: 1
    }, 
    'E11_postmr': {
        0: 0,
        1: 1
    }, 
    'E13_premr': {
        0: 0,
        1: 1
    }, 
    'E13_postmr': {
        0: 0,
        1: 1
    }, 
    'E14_premr': {
        0: 0,
        1: 1
    }, 
    'E14_postmr': {
        0: 0,
        1: 1
    },
}