from nnicotine.datasets.cath import _align_subseqs as _resolve_gaps

refseq = "ABCDEFGHI"
strucseq = refseq
result = _resolve_gaps(strucseq, refseq)
print(result[0])
print(refseq)
assert result[0] == refseq

strucseq = "---FGHI"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "-----FGHI"

strucseq = "A-----G"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "A-----G--"

strucseq = "AB-----EFG"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "AB--EFG--"

strucseq = "AB-D-E-FGHI"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "AB-DEFGHI"

strucseq = "ABCD--EFG-I"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "ABCDEFG-I"

strucseq = "B--CD-FG"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "-BCD-FG--"


strucseq = "H--QAEILLTLKLQQKLFADPRRISLLKHIALSGSISQGAKDAGISYKSAWDAINE-NQLSEHILVER---------AVLTRYGQRLIQLYDLLAQIQQKAFDVLSDDD"
refseq = "GSHMQAEILLTLKLQQKLFADPRRISLLKHIALSGSISQGAKDAGISYKSAWDAINEMNQLSEHILVERATGGKGGGGAVLTRYGQRLIQLYDLLAQIQQKAFDVLSDDDALPLNSLLAAISRFSLQTSARNQWFGTITARDHDDVQQHVDVLLADGKTRLKVAITAQSGARLGLDEGKEVLILLKAPWVGITQDEAVAQNADNQLPGIISHIERGAEQCEVLMALPDGQTLCATVPVNEATSLQQGQNVTAYFNADSVIIATLC"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "--H-QAEILLTLKLQQKLFADPRRISLLKHIALSGSISQGAKDAGISYKSAWDAINE-NQLSEHILVER---------AVLTRYGQRLIQLYDLLAQIQQKAFDVLSDDD-----------------------------------------------------------------------------------------------------------------------------------------------------------"

strucseq = "HMAS-------------------------------------------------------------------------------------------------------PQFSQQREEDIYRFLKDNGPQRALVIAQALGMRTAKDVNRDLYRMKSRHLLDMDEQSKAWTIY"
refseq = "HMASPQFSQQREEDIYRFLKDNGPQRALVIAQALGMRTAKDVNRDLYRMKSRHLLDMDEQSKAWTIYRWTIY"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "HMASPQFSQQREEDIYRFLKDNGPQRALVIAQALGMRTAKDVNRDLYRMKSRHLLDMDEQSKAWTIY-----"

strucseq = "SIIKIDLESKTPIYKQIADQIIELIAKGELKPGDKLPSIRELASMLGVNMLTVNKAYNYLVDEGFIVVQKRRYVVKSEV---WRNMLRVIIYRALAS"
refseq = "MTSIIKIDLESKTPIYKQIADQIIELIAKGELKPGDKLPSIRELASMLGVNMLTVNKAYNYLVDEGFIVVQKRRYVVKSEVRDESWRNMLRVIIYRALASNMSKDEIVNEINRVVSEVNSK"
result = _resolve_gaps(strucseq, refseq)
assert result[0] == "--SIIKIDLESKTPIYKQIADQIIELIAKGELKPGDKLPSIRELASMLGVNMLTVNKAYNYLVDEGFIVVQKRRYVVKSEV----WRNMLRVIIYRALAS---------------------"
