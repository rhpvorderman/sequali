- `secondary_alignment.bam`
  
  - [HG00276/HG00276.haplotagged.bam](https://42basepairs.com/download/s3/ont-open-data/pgx_as_2025.07/analysis/cohort_1/output/HG00276/HG00276.haplotagged.bam)
  - `samtools view -h HG00276.haplotagged.bam | head -n 1000 | samtools view -Obam,level=9 -o secondary_alignment.bam`
