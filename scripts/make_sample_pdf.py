"""Generate data/sample_docs/travel_policy.pdf with PyMuPDF (so the PDF path is exercised by real files).

Usage:  python scripts/make_sample_pdf.py
"""
from pathlib import Path

import pymupdf  # PyMuPDF

PAGES = [
    """ACME CORP - BUSINESS TRAVEL POLICY
Version 1.7, effective 1 April 2025

1. Booking
All business travel must be booked through the corporate travel portal. Flights must be booked at least 14 days in advance whenever possible. Bookings made outside the portal are not reimbursed unless the travel manager approved them in writing.

2. Air Travel
Economy class is required for flights shorter than 6 hours. Business class may be booked for flights longer than 6 hours with approval from a vice president. Employees must choose the lowest logical fare.""",
    """3. Accommodation
Hotel stays are limited to 180 USD per night in standard cities and 250 USD per night in high-cost cities such as New York, London and Tokyo. Hotels must be booked through the corporate travel portal.

4. Ground Transport
Employees should use public transport or ride sharing. Rental cars require approval from the direct manager and must be compact or mid-size. Personal vehicle mileage is reimbursed at 0.67 USD per mile.

5. Travel Insurance and Emergencies
All employees are covered by the corporate travel insurance while on business trips. In an emergency, employees must call the 24-hour travel assistance line at +1-555-0100 and notify their manager within 24 hours.""",
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "travel_policy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(54, 54, 540, 780), text, fontsize=11, fontname="helv")
    doc.save(out)
    doc.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
