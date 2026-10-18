# Data sources

The bundled `example.npz` contains processed GSE123813 BCC immune-cell expression, source-patient response labels, fixed split indices and binary association masks. The resources below provide the source datasets and annotations used in the MAGENet study. Access and reuse follow each source's terms.

## Single-cell transcriptomes

GEO series pages link sample annotations, supplementary matrices and available sequencing records.

| Dataset | Cancer / use | GEO page | Processed-file downloads |
|---|---|---|---|
| GSE120575 | Melanoma | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/) |
| GSE123813 | Basal and squamous cell carcinoma; BCC example | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123813) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE123nnn/GSE123813/suppl/) |
| GSE169246 | Triple-negative breast cancer | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169246) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE169nnn/GSE169246/suppl/) |
| GSE200996 | Head and neck squamous cell carcinoma | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200996) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200996/suppl/) |
| GSE207422 | Non-small-cell lung cancer | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE207422) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE207nnn/GSE207422/suppl/) |
| GSE236581 | Colorectal and duodenal carcinoma | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236581) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236581/suppl/) |
| GSE130157 | External transcriptomic evaluation | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130157) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130157/suppl/) |
| GSE205506 | External transcriptomic evaluation | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205506) | [Files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/GSE205506/suppl/) |

## Biological associations

| Resource | Contents | Download page |
|---|---|---|
| gutMGene | Microbe–metabolite associations | [Resource downloads](https://bio-computing.hrbmu.edu.cn/gutmgene/#/Resource) |
| STITCH | Chemical–protein associations for metabolite–gene connections | [STITCH downloads](http://stitch-db.org/cgi/download.pl) |
| PubChem | Compound identifier mapping | [Download services](https://pubchem.ncbi.nlm.nih.gov/docs/downloads) |

The example includes the processed masks and ordered identifiers for 445 genes, 205 metabolites and 1,908 microbes. The masks contain association support only, encoded as `0` or `1`.

## Independent microbiome cohorts

| Dataset | Use | Download page |
|---|---|---|
| PRJEB10878 | Colorectal cancer microbiome analyses | [ENA study and sequencing files](https://www.ebi.ac.uk/ena/browser/view/PRJEB10878) |
| PRJNA750659 | Microbiome associations with ICB response | [ENA study and sequencing files](https://www.ebi.ac.uk/ena/browser/view/PRJNA750659) |

Clinical outcomes and sample annotations are available through the corresponding source studies.

## Paired omics data

ColPortal, Esteban-Gil et al., *Scientific Data* (2019): [source article](https://doi.org/10.1038/s41597-019-0198-z).

| Data | Public record |
|---|---|
| Processed microbiome | [Figshare](https://doi.org/10.6084/m9.figshare.7799129.v2) |
| Raw microbiome | [Figshare](https://doi.org/10.6084/m9.figshare.7799123.v4) |
| Gene expression | [Figshare](https://doi.org/10.6084/m9.figshare.7783532.v2) |
| Clinical metadata | [Figshare](https://doi.org/10.6084/m9.figshare.7807799.v1) |

Yang et al., *Theranostics* (2019), “Integrated microbiome and metabolome analysis reveals a novel interplay between commensal bacteria and metabolites in colorectal cancer”: [supplementary material](https://www.thno.org/v09p4101.htm#SM0), [tables](https://www.thno.org/v09/p4101/thnov09p4101s2.xls) and [supplementary PDF](https://www.thno.org/v09/p4101/thnov09p4101s1.pdf).

## Additional data and annotations

| Resource | Use | Download / access |
|---|---|---|
| P-NET prostate cancer data | Structural-attention evaluation | [Data instructions](https://github.com/marakeby/pnet_prostate_paper#installation); [author-provided archive](https://drive.google.com/uc?id=17nssbdUylkyQY1ebtxsIw5UzTAd0zxWb&export=download) |
| ImmPort | Immune-gene reference set | [Gene lists](https://www.immport.org/shared/genelists) |
| InnateDB | Innate-immunity annotations | [InnateDB](https://www.innatedb.com/) |
| KEGG | Compound and pathway annotations | [Data API](https://www.kegg.jp/kegg/rest/keggapi.html) |
| MSigDB | Gene sets | [Downloads](https://www.gsea-msigdb.org/gsea/msigdb/) |
