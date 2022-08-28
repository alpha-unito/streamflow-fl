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

doc: python $aggregate_script [--model MODEL]...  > state_dict_model.pt

baseCommand: [python]

inputs:
  aggregate_script:
    type: File
    doc: The script containing the aggregation logic
    inputBinding:
      position: 1     
  input_models:
    type:
      type: array
      items: File
      inputBinding:
        prefix: --model
    doc: The trained models, serialised with torch.save()
    inputBinding:
      position: 2
outputs:
  output_model:
    type: File
    doc: The aggregated model, serialised with torch.save()
    outputBinding:
      glob: state_dict_model.pt
