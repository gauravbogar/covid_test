import pandas as pd


def con_rec_dec(counts, date, district):
    try:

        confirmed = str(
            counts[
                (counts["date"] == date) & (counts["district"] == district)
            ].confirmed.values[0]
        )
        recovered = str(
            counts[
                (counts["date"] == date) & (counts["district"] == district)
            ].recovered.values[0]
        )
        deceased = str(
            counts[
                (counts["date"] == date) & (counts["district"] == district)
            ].deceased.values[0]
        )
        return confirmed, recovered, deceased
    except Exception as e:
        # print("Please enter a valid date(yyyy-mm-dd) and/or district")
        print(e)


counts = pd.read_csv(r"assets\districts.csv")
a, b, c = con_rec_dec(counts, "2020-05-12", "Mumbai")
print(a, b, c)

