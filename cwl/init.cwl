#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool

$namespaces:
  s: https://schema.org/

$schemas:
 - https://schema.org/version/latest/schemaorg-current-http.rdf

s:author:
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-9290-2017
    s:email: mailto:iacopo.colonnelli@unito.it
    s:name: Iacopo Colonnelli
  - class: s:Person
    s:identifier: https://orcid.org/0000-0002-9513-6087
    s:email: mailto:bruno.casella@unito.it
    s:name: Bruno Casella
  - class: s:Person
    s:identifier: https://orcid.org/0000-0001-8788-0829
    s:email: mailto:marco.aldinucci@unito.it
    s:name: Marco Aldinucci

s:codeRepository: https://github.com/alpha-unito/streamflow-fl
s:dateCreated: "2022-08-28"
s:license: https://spdx.org/licenses/LGPL-3.0-only
s:programmingLanguage: Python

doc: python $init_script > state_dict_model.pt

baseCommand: [python]

inputs:
  init_script:
    type: File
    inputBinding:
      position: 1

outputs:
  output_model:
    type: File
    doc: The model's state_dict with initialised weights, serialised with torch.save()
    outputBinding:
      glob: state_dict_model.pt
