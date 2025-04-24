import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Cào dữ liệu từ WHO Oral Health Fact Sheet
url = "https://www.who.int/news-room/fact-sheets/detail/oral-health"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Lấy văn bản từ các thẻ <p> (chứa nội dung chính)
paragraphs = soup.find_all("p")
text = " ".join([para.get_text(strip=True) for para in paragraphs])

# Tạo PDF
pdf_file = "who_oral_health.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
story = [Paragraph(text, styles["Normal"])]
doc.build(story)
print(f"Đã tạo file {pdf_file}")