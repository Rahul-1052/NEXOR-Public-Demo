import pandas as pd

payers_lookup = pd.read_csv("../data/payers_lookup.csv")

new_payer_names = [
    "UnitedCare Alliance",
    "AetnaPoint Health",
    "Blue Horizon Shield",
    "HumanaCore Plans",
    "Elevance Community Health",
    "Kaiser Integrated Care",
    "Centene Access Network",
    "CignaCare Benefits",
    "Molina Community Plans",
    "GuideWell Regional Health"
]

payer_groups = [
    "Commercial",
    "Commercial",
    "Commercial",
    "Medicare",
    "Commercial",
    "Integrated",
    "Medicaid",
    "Commercial",
    "Medicaid",
    "Commercial"
]

payers_lookup["PAYER_NAME"] = new_payer_names[:len(payers_lookup)]
payers_lookup["PAYER_GROUP"] = payer_groups[:len(payers_lookup)]

payers_lookup.to_csv("../data/payers_lookup.csv", index=False)

print("✅ Updated payers_lookup.csv")
print(payers_lookup)