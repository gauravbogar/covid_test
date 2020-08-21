def con_rec_dec(counts, date, district):
    try:

        confirmed = counts[
            (counts["date"] == date) & (counts["district"] == district)
        ].confirmed.values[0]
        recovered = counts[
            (counts["date"] == date) & (counts["district"] == district)
        ].recovered.values[0]
        deceased = counts[
            (counts["date"] == date) & (counts["district"] == district)
        ].deceased.values[0]
        return confirmed, recovered, deceased
    except:
        print("Please enter a valid date(yyyy-mm-dd) and/or district")
