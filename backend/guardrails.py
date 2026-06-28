import re

PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore previous instructions",
    r"(?i)abaikan instruksi sebelumnya",
    r"(?i)system prompt bypass",
    r"(?i)you are now a",
    r"(?i)mulai sekarang kamu adalah",
    r"(?i)forget everything",
    r"(?i)lupakan semua",
    r"(?i)acting as",
    r"(?i)jailbreak",
    r"(?i)override system",
]

SCHOOL_TAMPERING_PATTERNS = [
    r"(?i)hack komputer",
    r"(?i)hack server",
    r"(?i)ubah nilai",
    r"(?i)ganti nilai",
    r"(?i)bocoran soal",
    r"(?i)kunci jawaban ulangan",
    r"(?i)manipulasi absen",
]

TOXIC_LANGUAGE_PATTERNS = [
    r"(?i)\banjing\b", r"(?i)\bbabi\b", r"(?i)\bbangsat\b", r"(?i)\btolol\b",
    r"(?i)\bbodoh\b", r"(?i)\bgoblok\b", r"(?i)\bkontol\b", r"(?i)\bmemek\b",
    r"(?i)\basu\b", r"(?i)\bbitch\b", r"(?i)\bfuck\b", r"(?i)\bbastard\b",
]

HARM_VIOLENCE_SARA_PATTERNS = [
    r"(?i)bunuh", r"(?i)tembak", r"(?i)bom", r"(?i)kekerasan", r"(?i)aniaya", r"(?i)tawuran", 
    r"(?i)perang", r"(?i)senjata", r"(?i)miras", r"(?i)alkohol", r"(?i)narkoba", r"(?i)sabu-sabu",
    r"(?i)\bsara\b", r"(?i)\bfitnah\b", r"(?i)hina\b\s+(?:agama|ras|suku|antargolongan)",
    r"(?i)bokep", r"(?i)porn"
]

def is_safe_query(query: str) -> tuple[bool, str]:
    if not query or not query.strip():
        return True, ""       
    query_lower = query.lower()

    all_safety_patterns = (
        PROMPT_INJECTION_PATTERNS + 
        SCHOOL_TAMPERING_PATTERNS + 
        TOXIC_LANGUAGE_PATTERNS + 
        HARM_VIOLENCE_SARA_PATTERNS
    )
    
    for pattern in all_safety_patterns:
        if re.search(pattern, query_lower):
            return False, (
                "Maaf, konteks yang kamu tanyakan tidak masuk ke dalam lingkup materi kecerdasan buatan. "
                "Ajukan pertanyaan yang sesuai, seperti: Apa pengertian dari Kecerdasan Buatan (Artificial Intelligence)?, "
                "Apa perbedaan antara Machine Learning dan Deep Learning?, atau Bagaimana cara kerja Jaringan Saraf Tiruan secara sederhana?"
            )       
    return True, ""