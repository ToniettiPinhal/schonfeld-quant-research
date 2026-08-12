# Data policy

This repository does not redistribute the source return files. Running the project
downloads three ZIP archives directly from the Kenneth R. French Data Library into
`data/raw/`, which is ignored by Git:

- 49 value-weighted U.S. industry portfolios;
- Fama-French three factors and the risk-free rate;
- Fama-French Momentum factor.

Each run records the source URLs, SHA-256 checksums, byte sizes, download timestamp,
and available `Last-Modified` headers in `reports/run_metadata.json`.

The Data Library explicitly notes that reconstructed historical returns can change
when CRSP revises its database. Results in the committed report use the CRSP 202606
vintage. Exact reproduction requires archives matching the recorded checksums;
re-running against a newer vintage is a replication on revised data, not a byte-for-byte
reproduction.

Source and construction details:

- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_mom_factor.html

The source data remain subject to their owners' terms and copyright. The MIT license
in this repository applies to the project code, not to downloaded third-party data.
