#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: Workflow

$namespaces:
  cwltool: "http://commonwl.org/cwltool#"
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

requirements:
  InlineJavascriptRequirement: {}
  SubworkflowFeatureRequirement: {}

inputs:
  aggregate_script: File
  epochs_per_round: int
  init_script: File
  mnist_dataset: Directory
  rounds: int
  svhn_dataset: Directory
  train_mnist: File
  train_svhn: File

outputs: {}
steps:
  init:
    in:
      init_script: init_script
    out: [output_model]
    run: init.cwl
  loop:
    in:
      aggregate_script: aggregate_script
      epochs_per_round: epochs_per_round
      input_model: init/output_model
      mnist_dataset: mnist_dataset
      round:
        default: 0
      rounds: rounds
      svhn_dataset: svhn_dataset
      train_mnist: train_mnist
      train_svhn: train_svhn
    out: [output_model]
    requirements:
      cwltool:Loop:
        loopWhen: $(inputs.round < inputs.rounds)
        loop:
          round:
            valueFrom: $(inputs.round + 1)
          input_model: output_model
        outputMethod: last
    run:
      class: Workflow
      inputs:
        aggregate_script: File
        epochs_per_round: int
        input_model: File
        mnist_dataset: Directory
        round: int
        svhn_dataset: Directory
        train_mnist: File
        train_svhn: File
      outputs:
        output_model:
          type: File
          outputSource:
            aggregate/output_model
      steps:
        train_mnist:
          in:
            dataset: mnist_dataset
            epochs_per_round: epochs_per_round
            first_round:
              valueFrom: $(inputs.round == 0)
            input_model: input_model
            round: round
            train_script: train_mnist
          out: [output_model]
          run: train.cwl
        train_svhn:
          in:
            dataset: svhn_dataset
            epochs_per_round: epochs_per_round
            first_round:
              valueFrom: $(inputs.round == 0)
            input_model: input_model
            round: round
            train_script: train_svhn
          out: [output_model]
          run: train.cwl
        aggregate:
          in:
            aggregate_script: aggregate_script
            input_models:
              source:
                - train_mnist/output_model
                - train_svhn/output_model
          out: [output_model]
          run: aggregate.cwl
  eval_mnist:
    in:
      dataset: mnist_dataset
      epochs_per_round:
        default: 1
      first_round:
        default: false
      input_model: loop/output_model
      train_script: train_mnist
    out: []
    run: train.cwl
  eval_svhn:
    in:
      dataset: svhn_dataset
      epochs_per_round:
        default: 1
      first_round:
        default: false
      input_model: loop/output_model
      train_script: train_svhn
    out: []
    run: train.cwl
