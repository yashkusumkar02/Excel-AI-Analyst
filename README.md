# 📊 Excel-AI-Analyst

**AI-Powered Spreadsheet Analysis Tool**  
*Automate data cleaning, visualization, and insights with natural language queries.*


---

## 🚀 Features

- ✅ **1-Click Data Cleaning** – Remove duplicates, handle missing values, and fix encoding issues.
- 📊 **Smart Visualizations** – Auto-generate bar charts, line graphs, and histograms from your spreadsheet.
- 💬 **Chat Interface** – Ask questions like _"Show sales trends"_ or _"Find outliers"_ using natural language.
- 🧠 **Gemini AI Integration** *(Optional)* – Advanced analysis with Google Gemini's powerful language model.
- 🔒 **Privacy-Focused** – Your data stays on your machine; nothing is uploaded externally.

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Excel-AI-Analyst.git
cd Excel-AI-Analyst

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
```

---

## 🛠️ Configuration

### 1. Google Gemini API Key (Optional for AI features)

Create a `.env` file in the root directory:

```ini
GOOGLE_API_KEY=your_api_key_here
```

---

## 📂 Project Structure

```
Excel-AI-Analyst/
├── app/
│   ├── agents/              # AI analysis core
│   │   ├── excel_agent.py
│   │   ├── file_utils.py
│   │   └── utils.py
│   └── main.py              # Streamlit UI
├── docs/                    # Screenshots/DEMOs
├── requirements.txt         # Dependencies
├── .env.example             # Sample env config
└── README.md
```


---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

