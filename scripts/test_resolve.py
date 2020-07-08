from nnicotine.datasets.cath import _resolve_gaps

refseq = "ABCDEFGHI"
strucseq = [x for x in refseq]
result = _resolve_gaps(strucseq, refseq)
assert result == refseq

strucseq = [x for x in "---FGHI"]
result = _resolve_gaps(strucseq, refseq)
assert result == "CDEFGHI"

strucseq = [x for x in "A-----G"]
result = _resolve_gaps(strucseq, refseq)
assert result in refseq

strucseq = [x for x in "AB-----EFG"]
result = _resolve_gaps(strucseq, refseq)
assert result == "ABCD---EFG"

strucseq = [x for x in "AB-D-E-FGHI"]
result = _resolve_gaps(strucseq, refseq)
assert result == "ABCD-E-FGHI"

strucseq = [x for x in "ABCD--EFG-I"]
result = _resolve_gaps(strucseq, refseq)
assert result == "ABCD--EFGHI"

strucseq = [x for x in "B--CD-FG"]
result = _resolve_gaps(strucseq, refseq)
assert result == "B--CDEFG"


strucseq = [x for x in "H--QAEILLTLKLQQKLFADPRRISLLKHIALSGSISQGAKDAGISYKSAWDAINE-NQLSEHILVER---------AVLTRYGQRLIQLYDLLAQIQQKAFDVLSDDD"]
refseq = "GSHMQAEILLTLKLQQKLFADPRRISLLKHIALSGSISQGAKDAGISYKSAWDAINEMNQLSEHILVERATGGKGGGGAVLTRYGQRLIQLYDLLAQIQQKAFDVLSDDDALPLNSLLAAISRFSLQTSARNQWFGTITARDHDDVQQHVDVLLADGKTRLKVAITAQSGARLGLDEGKEVLILLKAPWVGITQDEAVAQNADNQLPGIISHIERGAEQCEVLMALPDGQTLCATVPVNEATSLQQGQNVTAYFNADSVIIATLC"
result = _resolve_gaps(strucseq, refseq)
assert ''.join(result.split('-')) in refseq

strucseq = [x for x in "HMAS-------------------------------------------------------------------------------------------------------PQFSQQREEDIYRFLKDNGPQRALVIAQALGMRTAKDVNRDLYRMKSRHLLDMDEQSKAWTIY"]
refseq = "HMASPQFSQQREEDIYRFLKDNGPQRALVIAQALGMRTAKDVNRDLYRMKSRHLLDMDEQSKAWTIYRWTIY"
result = _resolve_gaps(strucseq, refseq)
assert ''.join(result.split('-')) in refseq

strucseq = [x for x in "SIIKIDLESKTPIYKQIADQIIELIAKGELKPGDKLPSIRELASMLGVNMLTVNKAYNYLVDEGFIVVQKRRYVVKSEV---WRNMLRVIIYRALAS"]
refseq = "MTSIIKIDLESKTPIYKQIADQIIELIAKGELKPGDKLPSIRELASMLGVNMLTVNKAYNYLVDEGFIVVQKRRYVVKSEVRDESWRNMLRVIIYRALASNMSKDEIVNEINRVVSEVNSK"
result = _resolve_gaps(strucseq, refseq)
assert ''.join(result.split('-')) in refseq
