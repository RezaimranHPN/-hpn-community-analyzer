import re
from datetime import datetime
from collections import Counter
from io import BytesIO
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HPN Community Analyzer", page_icon="💬", layout="wide")

st.title("💬 HPN Community Analyzer")
st.caption("Analisis aktivitas WhatsApp Group • MVP lokal berbasis file ekspor .txt")
st.info("Privasi: file diproses di sesi aplikasi ini. MVP ini tidak mengirim data ke layanan AI eksternal. Analisis kualitas dan spam bersifat heuristik, bukan penilaian pasti.")

# Common WhatsApp exports:
# [31/10/26, 10:22:10] Name: message
# 31/10/26, 10:22 - Name: message
# 10/31/26, 10:22 AM - Name: message
# 2026-10-31, 10:22 - Name: message
HEADER_PATTERNS = [
    re.compile(r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s(.*?):\s(.*)$'),
    re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s-\s(.*?):\s(.*)$'),
    re.compile(r'^(\d{4}-\d{1,2}-\d{1,2}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\s-\s(.*?):\s(.*)$'),
]
SYSTEM_PHRASES = [
    "messages and calls are end-to-end encrypted", "created group", "added", "removed",
    "changed the subject", "changed this group's icon", "changed the group description",
    "left", "joined using this group's invite link", "you're now an admin",
    "this message was deleted", "security code changed"
]
SPAM_PHRASES = [
    "forwarded many times", "klik link", "click here", "daftar sekarang", "promo terbatas",
    "beli sekarang", "hubungi wa", "bit.ly/", "tinyurl.com", "wa.me/"
]
HELPFUL_TERMS = [
    "terima kasih", "makasih", "semoga", "menurutku", "pengalaman", "aku pernah",
    "saya pernah", "caranya", "coba", "bisa dibantu", "jawabannya", "informasi",
    "bermanfaat", "setuju", "izin berbagi", "sharing", "rekomendasi", "tips"
]
QUESTION_MARKERS = ["?", "bagaimana", "gimana", "kenapa", "apakah", "ada yang tahu", "minta saran"]
LOW_SIGNAL = {"ok", "oke", "iya", "ya", "wkwk", "haha", "hehe", "amin", "👍", "🙏", "😂", "❤️", "siap", "noted"}

def parse_datetime(date_s, time_s):
    t = time_s.strip().replace(" a.m.", " AM").replace(" p.m.", " PM")
    candidates = [
        "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p", "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p"
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(f"{date_s} {t}", fmt)
        except ValueError:
            continue
    # Try AM/PM variants and normalize spaces
    for fmt in ("%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p", "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p"):
        try:
            return datetime.strptime(f"{date_s} {t.upper()}", fmt)
        except ValueError:
            pass
    return None

def parse_chat(raw_text):
    rows = []
    current = None
    for line in raw_text.splitlines():
        match = None
        for pat in HEADER_PATTERNS:
            match = pat.match(line)
            if match:
                break
        if match:
            if current:
                rows.append(current)
            date_s, time_s, sender, message = match.groups()
            dt = parse_datetime(date_s, time_s)
            current = {"datetime": dt, "date_raw": date_s, "time_raw": time_s,
                       "sender": sender.strip(), "message": message.strip()}
        elif current is not None:
            current["message"] += "\n" + line
    if current:
        rows.append(current)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["message"] = df["message"].fillna("").astype(str)
    df["sender"] = df["sender"].fillna("").astype(str)
    df["is_system"] = df["sender"].str.lower().isin(["", "system"]) | df["message"].str.lower().apply(
        lambda m: any(p in m for p in SYSTEM_PHRASES) and len(m) < 180
    )
    df["is_duplicate"] = False
    normalized = df["message"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["normalized"] = normalized
    df["is_duplicate"] = normalized.ne("") & normalized.duplicated(keep=False)
    df["is_spam_suspect"] = df["message"].str.lower().apply(
        lambda m: any(p in m for p in SPAM_PHRASES)
    ) | df["is_duplicate"]
    df["word_count"] = df["message"].str.findall(r"\b[\w'-]+\b").str.len()
    df["is_low_signal"] = normalized.apply(lambda m: m in LOW_SIGNAL or (len(m) <= 3 and not "?" in m))
    df["has_question"] = df["message"].str.lower().apply(lambda m: any(q in m for q in QUESTION_MARKERS))
    df["helpful_signal"] = df["message"].str.lower().apply(lambda m: any(t in m for t in HELPFUL_TERMS))
    df["is_substantive"] = (df["word_count"] >= 8) | df["has_question"] | df["helpful_signal"]
    return df

def make_summary(df):
    messages = df[(~df["is_system"]) & df["sender"].ne("")]
    if messages.empty:
        return pd.DataFrame()
    g = messages.groupby("sender", dropna=False)
    summary = g.agg(
        total_messages=("message", "size"),
        active_days=("datetime", lambda x: x.dropna().dt.date.nunique()),
        first_message=("datetime", "min"),
        last_message=("datetime", "max"),
        duplicate_messages=("is_duplicate", "sum"),
        spam_suspects=("is_spam_suspect", "sum"),
        low_signal_messages=("is_low_signal", "sum"),
        substantive_messages=("is_substantive", "sum"),
        questions=("has_question", "sum"),
        helpful_signals=("helpful_signal", "sum"),
        total_words=("word_count", "sum")
    ).reset_index().rename(columns={"sender": "member"})
    summary["nonspam_estimate"] = (summary["total_messages"] - summary["spam_suspects"]).clip(lower=0)
    summary["spam_rate_pct"] = (summary["spam_suspects"] / summary["total_messages"] * 100).round(1)
    # Transparent heuristic score, not a definitive AI judgment
    max_days = max(int(summary["active_days"].max()), 1)
    max_nonspam = max(int(summary["nonspam_estimate"].max()), 1)
    max_sub = max(int(summary["substantive_messages"].max()), 1)
    max_help = max(int(summary["helpful_signals"].max()), 1)
    consistency = summary["active_days"] / max_days * 100
    activity = summary["nonspam_estimate"] / max_nonspam * 100
    substance = summary["substantive_messages"] / max_sub * 100
    helpful = summary["helpful_signals"] / max_help * 100
    spam_penalty = (summary["spam_rate_pct"] * 0.5).clip(upper=35)
    summary["contribution_score"] = (0.25*activity + 0.25*consistency + 0.30*substance + 0.20*helpful - spam_penalty).clip(0, 100).round(1)
    summary["activity_rank"] = summary["nonspam_estimate"].rank(method="min", ascending=False).astype(int)
    summary["consistency_rank"] = summary["active_days"].rank(method="min", ascending=False).astype(int)
    summary["contribution_rank"] = summary["contribution_score"].rank(method="min", ascending=False).astype(int)
    summary = summary.sort_values(["contribution_rank", "nonspam_estimate"], ascending=[True, False])
    return summary

uploaded = st.file_uploader("Upload file ekspor WhatsApp (.txt)", type=["txt"])
if uploaded:
    try:
        raw = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
    df = parse_chat(raw)
    if df.empty:
        st.error("Belum berhasil membaca format file ini. Pastikan file berasal dari Export Chat WhatsApp dan memiliki baris berisi tanggal, waktu, serta nama pengirim.")
        st.code("Contoh format yang didukung:\n[31/10/26, 10:22:10] Nama Member: Isi pesan\n31/10/26, 10:22 - Nama Member: Isi pesan")
    else:
        senders = sorted(df.loc[~df["is_system"], "sender"].dropna().unique().tolist())
        if not senders:
            st.warning("Pesan terbaca, tetapi nama pengirim belum terdeteksi dengan baik.")
        else:
            summary = make_summary(df)
            start = df["datetime"].min()
            end = df["datetime"].max()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Member terdeteksi", f"{summary['member'].nunique():,}")
            c2.metric("Pesan member", f"{int(summary['total_messages'].sum()):,}")
            c3.metric("Pesan terindikasi spam/duplikat", f"{int(summary['spam_suspects'].sum()):,}")
            c4.metric("Rentang tanggal", f"{start:%d %b %Y} – {end:%d %b %Y}" if pd.notna(start) and pd.notna(end) else "Tidak diketahui")
            st.caption("Catatan: deteksi spam menggunakan aturan awal. Pesan yang sama dapat wajar dalam konteks tertentu; periksa sebelum mengambil keputusan.")
            st.subheader("Ranking member")
            ranking_type = st.selectbox("Urutkan ranking berdasarkan", ["Kontribusi keseluruhan", "Jumlah pesan nonspam (perkiraan)", "Hari aktif"])
            if ranking_type == "Jumlah pesan nonspam (perkiraan)":
                shown = summary.sort_values(["nonspam_estimate", "active_days"], ascending=False).copy()
            elif ranking_type == "Hari aktif":
                shown = summary.sort_values(["active_days", "nonspam_estimate"], ascending=False).copy()
            else:
                shown = summary.sort_values(["contribution_score", "nonspam_estimate"], ascending=False).copy()
            top_n = st.selectbox("Tampilkan", ["Semua member", "Top 10", "Top 20", "Top 50"])
            if top_n != "Semua member":
                shown = shown.head(int(top_n.split()[1]))
            display = shown[["member", "total_messages", "nonspam_estimate", "active_days", "spam_suspects", "spam_rate_pct", "substantive_messages", "helpful_signals", "contribution_score"]].rename(columns={
                "member":"Member", "total_messages":"Total Pesan", "nonspam_estimate":"Pesan Nonspam (perkiraan)",
                "active_days":"Hari Aktif", "spam_suspects":"Indikasi Spam/Duplikat",
                "spam_rate_pct":"Indikasi Spam (%)", "substantive_messages":"Pesan Substantif (heuristik)",
                "helpful_signals":"Sinyal Membantu", "contribution_score":"Skor Kontribusi"
            })
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.subheader("Ringkasan aktivitas")
            st.bar_chart(shown.set_index("member")["nonspam_estimate"].head(20))
            st.subheader("Rincian pesan")
            with st.expander("Lihat pesan yang terindikasi spam atau duplikat"):
                flagged = df[(~df["is_system"]) & df["is_spam_suspect"]][["datetime", "sender", "message", "is_duplicate"]].copy()
                st.dataframe(flagged, use_container_width=True, hide_index=True)
            with st.expander("Lihat seluruh pesan yang berhasil dibaca"):
                st.dataframe(df[["datetime", "sender", "message", "is_system"]], use_container_width=True, hide_index=True)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                summary.to_excel(writer, sheet_name="Ranking Lengkap", index=False)
                df.to_excel(writer, sheet_name="Pesan Terbaca", index=False)
                flagged = df[(~df["is_system"]) & df["is_spam_suspect"]][["datetime", "sender", "message", "is_duplicate"]]
                flagged.to_excel(writer, sheet_name="Periksa Spam", index=False)
            st.download_button("⬇️ Download laporan Excel", data=output.getvalue(),
                               file_name="HPN_Community_Analysis.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.markdown("### Cara menggunakan")
    st.markdown("1. Buka grup WhatsApp HPN.  \n2. Pilih **Ekspor chat → Tanpa media**.  \n3. Upload file `.txt` di atas.  \n4. Periksa ranking dan unduh laporan Excel.")
    st.markdown("### Indikator ranking")
    st.markdown("- **Aktivitas:** jumlah pesan setelah mengurangi pesan yang terindikasi spam/duplikat.  \n- **Konsistensi:** jumlah hari berbeda saat member mengirim pesan.  \n- **Kontribusi:** skor heuristik dari pesan substantif, sinyal membantu, aktivitas, konsistensi, dan penalti spam.")
    st.warning("Versi ini belum menggunakan model AI eksternal untuk memahami konteks secara mendalam. Skor kontribusi adalah estimasi berbasis aturan dan perlu ditinjau manusia.")
