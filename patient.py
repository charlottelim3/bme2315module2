import csv


class patient_objects:
    """Stores patient-level clinical data and supports CSV loading/filtering."""

    all_patients = []

    def __init__(
        self,
        age_of_symptom_onset,
        age_of_diagnosis,
        age_of_death,
        sex,
    ):
        self.age_of_symptom_onset = age_of_symptom_onset
        self.age_of_diagnosis = age_of_diagnosis
        self.age_of_death = age_of_death
        self.sex = sex
        patient_objects.all_patients.append(self)

    def __repr__(self):
        return (
            f"Patient: ({self.age_of_symptom_onset} | "
            f"{self.age_of_diagnosis} | {self.age_of_death} | {self.sex})"
        )

    def get_age_of_diagnosis(self):
        """Return diagnosis age; place missing values at the end when sorting."""
        if self.age_of_diagnosis == "":
            return float("inf")
        return self.age_of_diagnosis

    @classmethod
    def instantiate_from_csv(cls, filename):
        """Create patient objects from the original patient-demographics CSV."""
        cls.all_patients.clear()

        with open(filename, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row in reader:
                patient_objects(
                    convert_age(row["Age of onset cognitive symptoms"]),
                    convert_age(row["Age of Dementia diagnosis"]),
                    convert_age(row["Age at Death"]),
                    row["Sex"],
                )

    @classmethod
    def filter_age_range(
        cls, diagnosis_min, diagnosis_max, death_min, death_max
    ):
        """Filter patients by diagnosis-age and death-age ranges."""
        filtered_patients = []

        for patient in cls.all_patients:
            if (
                patient.age_of_diagnosis != ""
                and patient.age_of_death != ""
                and diagnosis_min <= patient.age_of_diagnosis <= diagnosis_max
                and death_min <= patient.age_of_death <= death_max
            ):
                filtered_patients.append(patient)

        return filtered_patients


def convert_age(value):
    """Keep missing ages as empty strings and convert recorded ages to integers."""
    if value == "":
        return ""
    return int(float(value))
