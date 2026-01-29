from typing import Optional, Literal, List
import pandas as pd
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator

tumor_info_mapping = {
    'Eingriff': 'intervention',
    'Dignität': 'dignity',
    'Entität': 'entity',
    'Subentität': 'subentity',
    'Größe': 'size',
    'Lokalisation': 'location',
    'Grading': 'grading',
    'Resektion': 'resection_margin',
    'Regression': 'regression',
    'Prefix': 'TNM_prefix',
    'T': 'TNM_T',
    'N': 'TNM_N',
    'M': 'TNM_M',
    'V': 'TNM_V',
    'L': 'TNM_L',
    'Pn': 'TNM_Pn'
}


class TumorTextSequence(BaseModel):
    sequence: str = Field(min_length=2, description='Text sequence containing the tumor information')
    type: Literal['intervention', 'dignity', 'entity', 'subentity', 'size', 'location',
                  'grading', 'resection_margin', 'regression', 'TNM_prefix', 'TNM_T',
                  'TNM_N', 'TNM_M', 'TNM_V', 'TNM_L', 'TNM_Pn'] = Field(description='Type of the tumor information')
    start: Optional[int] = Field(default=0, ge=0, description='Start position of the sequence in the text')
    end: Optional[int] = Field(default=0, ge=0, description='End position of the sequence in the text')


class TumorClassification(BaseModel):
    # retrieved_text: List[TumorTextSequence]
    intervention: Literal['Biopsie', 'Curettage', 'Resektion']
    dignity: Literal['benigne', 'maligne', 'unknown']
    entity: str = Field(..., description='Name of the tumor entity')
    subentity: Optional[str] = Field(None, description='Name of the subentity')
    size: Optional[float] = Field(None, gt=0, description='Size of the tumor in cm')
    location: Optional[str] = Field(None, description='Location of the tumor')
    grading: Optional[Literal['G1', 'G2', 'G3', 'G4', 'GX']] = None
    resection_margin: Optional[Literal['R0', 'R1', 'R2', 'RX']] = None
    regression: Optional[Literal['Rg1', 'Rg2', 'Rg3', 'Rg4', 'Rg5', 'Rg6','RgX']] = None
    TNM_prefix: Optional[Literal['a', 'c', 'p', 'u', 'y', 'r', 'm', 'X']] = None
    TNM_T: Optional[Literal['T0', 'T1', 'T2', 'T3', 'T4', 'TX']] = None
    TNM_N: Optional[Literal['N0', 'N1', 'NX']] = None
    TNM_M: Optional[Literal['M0', 'M1', 'MX']] = None
    TNM_V: Optional[Literal['V0', 'V1', 'V2', 'VX']] = None
    TNM_L: Optional[Literal['L0', 'L1', 'LX']] = None
    TNM_Pn: Optional[Literal['Pn0', 'Pn1', 'PnX']] = None
    

    @model_validator(mode='after')
    def validate_tumor_classes(self) -> Self:
        if self.intervention != 'resection':
            assert self.size == None, 'Size is only relevant for resected tumors'
            assert self.resection_margin == None, 'Resection margin is only relevant for resected tumors'
            assert self.regression == None, 'Regression is only relevant for resected tumors'
            assert self.TNM_T == None, 'TNM_T classification is only relevant for resected tumors'

        if self.dignity == 'benign':
            assert self.grading == None, 'Benign tumors do not have grading'
            assert self.regression == None, 'Benign tumors do not have regression'
            assert self.TNM_prefix == None, 'Benign tumors do not have TNM classification'
            assert self.TNM_N == None, 'Benign tumors do not have TNM classification'
            assert self.TNM_T == None, 'Benign tumors do not have TNM classification'
            assert self.TNM_M == None, 'Benign tumors do not have TNM classification'
            assert self.TNM_V == None, 'Benign tumors do not have TNM classification'
            assert self.TNM_L == None, 'Benign tumors do not have TNM classification'
            assert self.TNM_Pn == None, 'Benign tumors do not have TNM classification'

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(
            retrieved_text=[TumorTextSequence(sequence=row['text'], 
                type=tumor_info_mapping[row['labels']], start=row['start'], 
                end=row['end']) for row in df['label']],
            intervention=df['eingriff'],
            dignity=df['dignitaet'],
            entity=df['entitat'],
            subentity=df['subentitat'],
            size=df['groesse'],
            location=df['lokalisation'],
            grading=df['grading'],
            resection_margin=df['resektion'],
            regression=df['regression'],
            TNM_prefix=df['Prefix'],
            TNM_T=df['T'],
            TNM_N=df['N'],
            TNM_M=df['M'],
            TNM_V=df['V'],
            TNM_L=df['L'],
            TNM_Pn=df['Pn']
        )


class TumorClassificationSimple(BaseModel):
    # retrieved_text: List[TumorTextSequence]
    intervention: Literal['Biopsie', 'Curettage', 'Resektion']
    dignity: Literal['benigne', 'maligne', 'unknown'] = Field("unknown")
    entity: str = Field(..., description='Name of the tumor entity')
    subentity: Optional[str] = Field(None, description='Name of the subentity')
    # size: Optional[float] = Field(None, gt=0, description='Size of the tumor in cm')
    location: Optional[str] = Field(None, description='Location of the tumor')

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(
            intervention=df['eingriff'],
            dignity=df['dignitaet'] if df['dignitaet'] is not None else 'unknown',
            entity=df['entitat'],
            subentity=df['subentitat'],
            # size=df['groesse'],
            location=df['lokalisation'],
        )


TumorClassificationSimple.model_json_schema()
TumorClassification.model_json_schema()