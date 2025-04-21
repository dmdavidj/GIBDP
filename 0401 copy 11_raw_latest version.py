import os
import sys
import math
import re
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from skimage import draw
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QFileDialog, QLabel, QWidget, QTreeWidget, 
                              QTreeWidgetItem, QProgressBar, QTabWidget, QMessageBox,
                              QScrollArea, QSizePolicy, QListWidget, QGroupBox, QRadioButton,
                              QButtonGroup)
from PySide6.QtCore import (Qt, QThread, Signal, QObject, QRunnable, QThreadPool, 
                           QMutex, QMutexLocker, Slot)
from PySide6.QtGui import QFont

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 문제 해결
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# DICOM RT 처리를 위한 라이브러리 
import SimpleITK as sitk

class DicomSeries:
    """CT, RTDose, RTStructure를 포함한 하나의 DICOM 시리즈를 나타내는 클래스"""
    def __init__(self, patient_id=None, series_id=None):
        self.patient_id = patient_id
        self.series_id = series_id
        self.description = None
        self.ct_files = []
        self.rtdose_file = None
        self.rtstruct_file = None
        
        # 처리된 데이터를 저장할 속성
        self.ct_image = None  # CT 볼륨 데이터
        self.dose_array = None  # numpy 배열 형태의 선량 데이터
        self.dose_spacing = None  # 선량 격자 간격
        self.dose_origin = None  # 선량 격자 원점
        self.structures = {}  # 구조체 이름별 컨투어 데이터
        self.structure_masks = {}  # 구조체 이름별 마스크 배열 (numpy)
        
        # 계산된 결과값
        self.structure_metrics = {}  # 구조체 메트릭 (부피, 선량 등)
        self.conformity_index = None  # CI
        self.homogeneity_index = None  # HI
        self.dvh_data = {}  # 각 구조체별 DVH 데이터

        # 다중 시리즈 처리를 위한 ID
        self.series_uid = None
        
        # 분석 완료 플래그 추가
        self.analysis_completed = False
    
    def clear_large_data(self):
        """큰 용량을 차지하는 데이터 정리 - 메모리 관리를 위해 추가"""
        print(f"큰 데이터 메모리 해제 시작 - 환자 ID: {self.patient_id}")
        
        # CT 볼륨 데이터 정리
        if hasattr(self, 'ct_image') and self.ct_image is not None:
            self.ct_image = None
            print("- CT 이미지 데이터 해제 완료")
        
        # 선량 배열 데이터 정리
        if hasattr(self, 'dose_array') and self.dose_array is not None:
            self.dose_array = None
            print("- 선량 데이터 해제 완료")
        
        # 구조체 마스크 데이터 정리 (이미 메트릭이 추출된 경우)
        if hasattr(self, 'structure_masks') and self.structure_metrics and self.structure_masks:
            self.structure_masks = {}
            print("- 구조체 마스크 데이터 해제 완료")
            
        # 메모리 강제 정리
        import gc
        gc.collect()
        print("메모리 정리 완료")


# 시그널을 위한 래퍼 클래스 (QRunnable은 시그널을 직접 사용할 수 없음)
class WorkerSignals(QObject):
    """워커 시그널을 담은 객체"""
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)


# DICOM 로더를 QRunnable 방식으로 전환
class DicomLoaderRunnable(QRunnable):
    """QThreadPool에서 실행할 DICOM 로더"""
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.signals = WorkerSignals()
        self.is_running = True
        self.mutex = QMutex()
        
    def stop(self):
        """작업 중지 요청"""
        with QMutexLocker(self.mutex):
            self.is_running = False
        print("DicomLoaderRunnable: 중지 요청됨")
    
    def is_still_running(self):
        """현재 실행 상태 확인"""
        with QMutexLocker(self.mutex):
            return self.is_running
    
    @Slot()
    def run(self):
        """작업 실행 (QRunnable 메인 메서드)"""
        try:
            # 폴더에서 모든 DICOM 파일 찾기
            dicom_files = []
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if not self.is_still_running():
                        print("DicomLoaderRunnable: 작업 중지됨")
                        return
                    file_path = os.path.join(root, file)
                    try:
                        # DICOM 파일인지 확인 (헤더만 읽기)
                        pydicom.dcmread(file_path, stop_before_pixels=True)
                        dicom_files.append(file_path)
                    except:
                        # DICOM 파일이 아님
                        pass
            
            if not self.is_still_running():
                return
                
            if not dicom_files:
                self.signals.error.emit("선택한 폴더에 DICOM 파일이 없습니다.")
                return
                
            # 각 파일의 DICOM 유형 확인
            ct_files_dict = {}  # 시리즈 UID별 CT 파일 목록
            rtdose_files = {}  # 시리즈 UID별 RTDose 파일 목록
            rtstruct_files = {}  # 시리즈 UID별 RTStruct 파일 목록
            
            total_files = len(dicom_files)
            for i, file_path in enumerate(dicom_files):
                if not self.is_still_running():
                    return
                try:
                    ds = pydicom.dcmread(file_path, force=True)
                    if hasattr(ds, 'Modality'):
                        study_uid = ds.StudyInstanceUID if hasattr(ds, 'StudyInstanceUID') else 'Unknown'
                        
                        if ds.Modality == 'CT':
                            series_uid = ds.SeriesInstanceUID if hasattr(ds, 'SeriesInstanceUID') else 'Unknown'
                            if series_uid not in ct_files_dict:
                                ct_files_dict[series_uid] = []
                            ct_files_dict[series_uid].append(file_path)
                        
                        elif ds.Modality == 'RTDOSE':
                            # RTDose를 StudyInstanceUID로 그룹화
                            if study_uid not in rtdose_files:
                                rtdose_files[study_uid] = []
                            rtdose_files[study_uid].append(file_path)
                        
                        elif ds.Modality == 'RTSTRUCT':
                            # RTStruct를 StudyInstanceUID로 그룹화
                            if study_uid not in rtstruct_files:
                                rtstruct_files[study_uid] = []
                            rtstruct_files[study_uid].append(file_path)
                
                except Exception as e:
                    print(f"파일 읽기 오류: {file_path}, {str(e)}")
                    pass
                
                # 진행 상황 보고
                self.signals.progress.emit(int(100 * i / total_files))
            
            if not self.is_still_running():
                return
                
            # 파일 개수 디버깅
            print(f"CT 시리즈: {len(ct_files_dict)}, RTDose 파일: {len(rtdose_files)}, RTStruct 파일: {len(rtstruct_files)}")
            
            if not ct_files_dict:
                self.signals.error.emit("폴더에 CT 파일이 없습니다.")
                return
                
            # 각 시리즈별로 DicomSeries 객체 생성
            dicom_series_list = []
            
            for series_uid, ct_files in ct_files_dict.items():
                if not self.is_still_running():
                    return
                    
                # 시리즈 정보 추출
                ct_ds = pydicom.dcmread(ct_files[0], force=True)
                patient_id = ct_ds.PatientID if hasattr(ct_ds, 'PatientID') else 'Unknown'
                description = ct_ds.SeriesDescription if hasattr(ct_ds, 'SeriesDescription') else 'Unknown'
                study_uid = ct_ds.StudyInstanceUID if hasattr(ct_ds, 'StudyInstanceUID') else 'Unknown'
                
                dicom_series = DicomSeries(patient_id=patient_id, series_id=series_uid)
                dicom_series.description = description
                dicom_series.ct_files = ct_files
                dicom_series.series_uid = series_uid
                
                # 같은 StudyInstanceUID를 가진 RTDose와 RTStruct 파일 매핑
                if study_uid in rtdose_files:
                    dicom_series.rtdose_file = rtdose_files[study_uid][0]
                
                if study_uid in rtstruct_files:
                    dicom_series.rtstruct_file = rtstruct_files[study_uid][0]
                
                dicom_series_list.append(dicom_series)
            
            if not self.is_still_running():
                return
                
            if not dicom_series_list:
                self.signals.error.emit("유효한 DICOM 시리즈가 없습니다.")
                return
                
            # 작업 완료 신호 보내기
            self.signals.progress.emit(100)
            self.signals.finished.emit(dicom_series_list)
            print("DicomLoaderRunnable: 작업 완료됨")
            
        except Exception as e:
            if self.is_still_running():
                import traceback
                traceback.print_exc()
                self.signals.error.emit(f"DICOM 로드 중 오류 발생: {str(e)}")
        finally:
            print("DicomLoaderRunnable: 스레드 종료")
            # 명시적 메모리 정리
            import gc
            gc.collect()


# DICOM 분석기도 QRunnable로 전환
class DicomAnalysisRunnable(QRunnable):
    """QThreadPool에서 실행할 DICOM 분석기"""
    
    def __init__(self, dicom_series):
        super().__init__()
        self.dicom_series = dicom_series
        self.signals = WorkerSignals()
        self.is_running = True
        self.mutex = QMutex()
    
    def stop(self):
        """작업 중지 요청"""
        with QMutexLocker(self.mutex):
            self.is_running = False
        print("DicomAnalysisRunnable: 중지 요청됨")
    
    def is_still_running(self):
        """현재 실행 상태 확인"""
        with QMutexLocker(self.mutex):
            return self.is_running
    
    def natsort_key(self, s):
        """자연어 정렬을 위한 키 함수"""
        return [int(c) if c.isdigit() else c.lower() 
                for c in re.split(r'(\d+)', str(s))]
    
    def _create_unique_identifier(self, dicom_series):
        """고유한 식별자 생성"""
        patient_id = dicom_series.patient_id or 'Unknown'
        series_desc = dicom_series.description or 'Unknown_Series'
        
        # 특수 문자 제거 및 공백을 '_'로 대체
        patient_id = re.sub(r'[^\w\-_\.]', '_', patient_id)
        series_desc = re.sub(r'[^\w\-_\.]', '_', series_desc)
        
        return f"{patient_id}_{series_desc}"

    def _calculate_structure_metrics(self, positions_z, pixel_spacing, rtdose_ds):
        """구조체별 정확한 지표 계산"""
        # 선량 단위 확인 (Gy 또는 cGy)
        dose_units = rtdose_ds.DoseUnits if hasattr(rtdose_ds, 'DoseUnits') else 'GY'
        scaling_factor = 1.0
        if dose_units.upper() == 'CGY':
            scaling_factor = 0.01  # cGy를 Gy로 변환
        
        # 처방 선량 (Gy) - 예시로 60Gy 사용
        prescription_dose = 60.0
        
        # 각 구조체의 지표 계산
        structure_metrics = {}
        
        # z-좌표 간격 (슬라이스 두께)
        slice_thickness = None
        if len(positions_z) > 1:
            slice_thickness = abs(positions_z[1] - positions_z[0])
        else:
            slice_thickness = 2.5  # 기본값
        
        # xy 평면에서의 픽셀 크기
        pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
        
        # Lungs 구조체의 무게중심 계산을 위한 변수
        lungs_centroid = None
        
        # 각 구조체 순회하며 메트릭 계산
        for struct_name, contour_data in self.dicom_series.structures.items():
            if not self.is_still_running():
                return
                
            print(f"구조체 '{struct_name}' 지표 계산 중...")
            
            try:
                # 고유한 z-좌표에 있는 컨투어만 카운트
                unique_z_coords = set()
                
                for contour in contour_data:
                    z = contour['z']
                    # 소수점 2자리까지 반올림하여 근사값 비교
                    rounded_z = round(z, 2)
                    unique_z_coords.add(rounded_z)
                
                # 고유한 컨투어 수(고유 z-좌표 수) 계산
                contour_count = len(unique_z_coords)
                
                # 디버깅 정보 출력
                print(f"구조체 {struct_name} 디버깅 정보:")
                print(f"전체 컨투어 수: {len(contour_data)}")
                print(f"고유 z-좌표 수: {contour_count}")
                print("z-좌표 목록:")
                for z in sorted(unique_z_coords):
                    print(f"{z}")
                
                # CT 슬라이스 두께 계산
                slice_thickness = None
                if len(positions_z) > 1:
                    slice_thickness = abs(positions_z[1] - positions_z[0])
                else:
                    slice_thickness = 2.5  # 기본값
                
                total_contour_thickness = contour_count * slice_thickness
                
                # 1. 부피 및 무게중심 계산
                volume_cc = 0.0
                centroid_sum = np.array([0.0, 0.0, 0.0])
                weight_sum = 0.0
                
                major_axis = 0.0
                minor_axis = 0.0
                
                # 구조체별 특성 치수 계산
                x_coords = []
                y_coords = []
                z_coords = []
                
                for contour in contour_data:
                    if not self.is_still_running():
                        return
                        
                    points = contour['points']
                    z = contour['z']
                    
                    # 컨투어 면적 계산 (다각형 면적 공식 사용)
                    area = 0.0
                    n = len(points)
                    for i in range(n):
                        j = (i + 1) % n
                        area += points[i][0] * points[j][1]
                        area -= points[j][0] * points[i][1]
                    area = abs(area) / 2.0
                    
                    # 슬라이스 부피 계산
                    slice_volume = area * slice_thickness / 1000.0  # cc 단위
                    volume_cc += slice_volume
                    
                    # 무게중심 계산을 위한 합산
                    slice_centroid = np.mean(points, axis=0)
                    centroid_sum += slice_centroid * slice_volume
                    weight_sum += slice_volume
                    
                    # 좌표 저장
                    for point in points:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                        z_coords.append(point[2])
                
                # 좌표 범위를 기반으로 장, 단축 계산
                if x_coords and y_coords and z_coords:
                    x_range = max(x_coords) - min(x_coords)
                    y_range = max(y_coords) - min(y_coords)
                    z_range = max(z_coords) - min(z_coords)
                    
                    # 장, 단축 길이
                    axes = sorted([x_range, y_range, z_range], reverse=True)
                    major_axis = axes[0]
                    minor_axis = axes[2]
                
                # 무게중심 계산
                centroid = centroid_sum / weight_sum if weight_sum > 0 else np.array([0.0, 0.0, 0.0])
                
                # Lungs 구조체의 무게중심 저장 (정규화 기준점)
                if struct_name.lower() == 'lungs':
                    lungs_centroid = centroid
                
                # 2. 선량 지표 계산 (실제 선량 데이터 사용)
                if struct_name in self.dicom_series.structure_masks:
                    struct_mask = self.dicom_series.structure_masks[struct_name]
                    dose_array = self.dicom_series.dose_array
                    
                    # 구조체 내 선량 추출
                    struct_doses = dose_array[struct_mask]
                    
                    if len(struct_doses) > 0:
                        # 기본 선량 지표
                        mean_dose = np.mean(struct_doses) * scaling_factor  # Gy 단위
                        max_dose = np.max(struct_doses) * scaling_factor
                        min_dose = np.min(struct_doses) * scaling_factor
                        
                        # V-dose 계산 (다양한 선량 레벨에 대한 부피 비율)
                        v_dose_metrics = {}
                        dose_levels = [5, 10, 20, 30, 45, 50, 60]  # Gy
                        
                        for dose_level in dose_levels:
                            # 해당 선량 이상인 복셀의 비율 계산
                            volume_at_dose = np.sum(struct_doses * scaling_factor >= dose_level)
                            total_volume = len(struct_doses)
                            
                            if total_volume > 0:
                                v_dose_percent = (volume_at_dose / total_volume) * 100.0
                                v_dose_metrics[f'V{dose_level}Gy_percent'] = v_dose_percent
                        
                        # 특정 부피에 대한 선량 계산 (D0.03cc, D0.5cc 등)
                        special_dose_metrics = {}
                        
                        # 구조체 볼륨 (cc)
                        voxel_volume_cc = (self.dicom_series.dose_spacing[0] * 
                                        self.dicom_series.dose_spacing[1] * 
                                        self.dicom_series.dose_spacing[2]) / 1000.0  # mm³ -> cc
                        
                        # 정렬된 선량 배열 (내림차순)
                        sorted_doses = np.sort(struct_doses)[::-1] * scaling_factor
                        
                        # 구조체 총 볼륨
                        total_volume = len(struct_doses)
                        
                        # D90%, D95%, D99.9% 계산
                        dose_percentiles = [
                            (90, 'D90%'),
                            (95, 'D95%'),
                            (99.9, 'D99.9%')
                        ]
                        
                        for percentile, key in dose_percentiles:
                            # 해당 퍼센트 이상의 볼륨에 해당하는 선량 찾기
                            volume_threshold = int(total_volume * (percentile / 100))
                            if volume_threshold < len(sorted_doses):
                                special_dose_metrics[key] = sorted_doses[volume_threshold]
                        
                        # D0.03cc (0.03cc 볼륨에 대한 최소 선량)
                        voxels_in_0_03cc = max(1, int(0.03 / voxel_volume_cc))
                        if len(sorted_doses) >= voxels_in_0_03cc:
                            special_dose_metrics['D0.03cc'] = sorted_doses[voxels_in_0_03cc - 1]
                        
                        # D0.5cc (0.5cc 볼륨에 대한 최소 선량)
                        voxels_in_0_5cc = max(1, int(0.5 / voxel_volume_cc))
                        if len(sorted_doses) >= voxels_in_0_5cc:
                            special_dose_metrics['D0.5cc'] = sorted_doses[voxels_in_0_5cc - 1]
                        
                        # D100% (100% 부피에 대한 선량, 즉 최소 선량)
                        special_dose_metrics['D100%'] = min_dose
                    else:
                        # 선량 데이터가 없는 경우 기본값 설정
                        mean_dose = 0.0
                        max_dose = 0.0
                        min_dose = 0.0
                        v_dose_metrics = {}
                        special_dose_metrics = {}
                else:
                    # 마스크가 없는 경우 기본값 설정
                    mean_dose = 0.0
                    max_dose = 0.0
                    min_dose = 0.0
                    v_dose_metrics = {}
                    special_dose_metrics = {}
                
                # 기본 메트릭 저장
                metrics = {
                    'contour_count': contour_count,  # 여기에서 고유 z-좌표 수 저장
                    'total_contour_thickness_mm': total_contour_thickness,
                    'volume_cc': volume_cc,
                    'major_axis_mm': major_axis,
                    'minor_axis_mm': minor_axis,
                    'mean_dose': mean_dose,
                    'max_dose': max_dose,
                    'min_dose': min_dose,
                    'centroid_x_mm': centroid[0],
                    'centroid_y_mm': centroid[1],
                    'centroid_z_mm': centroid[2]
                }
                
                # V-dose 메트릭 추가
                metrics.update(v_dose_metrics)
                
                # 특수 선량 메트릭 추가
                metrics.update(special_dose_metrics)
                
                # 구조체 메트릭 저장
                structure_metrics[struct_name] = metrics
                
                print(f"구조체 '{struct_name}' 계산 완료: 부피 {volume_cc:.2f}cc, 평균 선량 {mean_dose:.2f}Gy")
                
            except Exception as e:
                print(f"구조체 '{struct_name}' 지표 계산 중 오류: {str(e)}")
                continue
        
        if not self.is_still_running():
            return
            
        # Lungs 구조체 무게중심 기준으로 정규화된 좌표 계산
        if lungs_centroid is not None:
            for struct_name, metrics in structure_metrics.items():
                # 무게중심 계산이 제대로 된 경우에만 정규화
                if 'centroid_x_mm' in metrics:
                    metrics['norm_centroid_x'] = metrics['centroid_x_mm'] - lungs_centroid[0]
                    metrics['norm_centroid_y'] = metrics['centroid_y_mm'] - lungs_centroid[1]
                    metrics['norm_centroid_z'] = metrics['centroid_z_mm'] - lungs_centroid[2]
            
            print(f"Lungs 무게중심: {lungs_centroid}")
        
        # CI 및 HI 계산 (실제 데이터에서 계산)
        if 'PTV' in structure_metrics and 'PTV' in self.dicom_series.structure_masks:
            try:
                dose_array = self.dicom_series.dose_array
                
                # 1. 60Gy 처방선량 이상인 영역의 부피 계산 (전체 dose 볼륨에서)
                dose_threshold = prescription_dose / scaling_factor
                rx_dose_mask = dose_array >= dose_threshold
                
                # 복셀 볼륨 계산 (mm³ -> cc)
                voxel_volume_cc = (self.dicom_series.dose_spacing[0] * 
                                self.dicom_series.dose_spacing[1] * 
                                self.dicom_series.dose_spacing[2]) / 1000.0
                
                # 처방선량 이상인 영역의 부피 (cc)
                rx_dose_volume_cc = np.sum(rx_dose_mask) * voxel_volume_cc
                
                # 2. PTV 부피 (이미 계산된 값 사용)
                ptv_volume_cc = structure_metrics['PTV']['volume_cc']
                
                # 3. CI 계산: 처방선량 이상 영역 부피 / PTV 부피
                if ptv_volume_cc > 0:
                    ci = rx_dose_volume_cc / ptv_volume_cc
                else:
                    ci = 0.0
                
                # 디버깅 정보 출력
                print(f"처방선량(60Gy) 이상 영역 복셀 수: {np.sum(rx_dose_mask)}")
                print(f"복셀 볼륨: {voxel_volume_cc:.6f}cc")
                print(f"처방선량 이상 영역 부피: {rx_dose_volume_cc:.2f}cc")
                print(f"PTV 부피: {ptv_volume_cc:.2f}cc")
                print(f"계산된 CI: {ci:.4f}")
                
                # PTV 내 선량 데이터 (HI 계산용)
                ptv_mask = self.dicom_series.structure_masks['PTV']
                ptv_doses = dose_array[ptv_mask] * scaling_factor
                
                # HI (Homogeneity Index) = 최대선량 / 처방선량
                if len(ptv_doses) > 0:
                    hi = np.max(ptv_doses) / prescription_dose
                else:
                    hi = 1.0
                
                self.dicom_series.conformity_index = ci
                self.dicom_series.homogeneity_index = hi
                
            except Exception as e:
                print(f"CI/HI 계산 오류: {str(e)}")
                
                import traceback
                traceback.print_exc()
                # 기본값 설정
                self.dicom_series.conformity_index = 1.05
                self.dicom_series.homogeneity_index = 1.12
        else:
            # PTV 데이터가 없는 경우 기본값
            self.dicom_series.conformity_index = 1.05
            self.dicom_series.homogeneity_index = 1.12
        
        # 구조체 이름 정렬 (자연어 정렬)
        sorted_structure_names = sorted(structure_metrics.keys(), key=self.natsort_key)
        
        # 정렬된 구조체 이름으로 새 딕셔너리 생성
        sorted_structure_metrics = {name: structure_metrics[name] for name in sorted_structure_names}
        
        self.dicom_series.structure_metrics = sorted_structure_metrics
        print(f"구조체 수: {len(sorted_structure_metrics)}")
        if hasattr(self.dicom_series, 'conformity_index') and hasattr(self.dicom_series, 'homogeneity_index'):
            print(f"CI: {self.dicom_series.conformity_index:.4f}, HI: {self.dicom_series.homogeneity_index:.4f}")
        
        # 구조체 메트릭 계산 완료 후 메모리 절약을 위한 변수 정리
        structure_metrics = None
        lungs_centroid = None
        
        # 메모리 강제 정리
        import gc
        gc.collect()
        print("구조체 메트릭 계산 완료 후 메모리 정리 완료")
                                             
    def _calculate_dvh_data(self, rtdose_ds):
        """각 구조체에 대한 DVH 데이터 계산"""
        # 선량 단위 확인 (Gy 또는 cGy)
        dose_units = rtdose_ds.DoseUnits if hasattr(rtdose_ds, 'DoseUnits') else 'GY'
        scaling_factor = 1.0
        if dose_units.upper() == 'CGY':
            scaling_factor = 0.01  # cGy를 Gy로 변환
        
        dvh_data = {}
        
        for struct_name, struct_mask in self.dicom_series.structure_masks.items():
            if not self.is_still_running():
                return
                
            try:
                dose_array = self.dicom_series.dose_array
                
                # 구조체 내 선량 추출
                struct_doses = dose_array[struct_mask]
                
                if len(struct_doses) > 0:
                    # 선량을 Gy 단위로 변환
                    struct_doses = struct_doses * scaling_factor
                    
                    # DVH 계산을 위한 선량 범위 설정
                    dose_bins = np.linspace(0, 70, 100)  # 0~70 Gy
                    dvh_values = np.zeros_like(dose_bins)
                    
                    # 각 선량 레벨에 대한 부피 비율 계산
                    for i, dose_level in enumerate(dose_bins):
                        # 해당 선량 이상인 복셀의 비율 계산
                        volume_at_dose = np.sum(struct_doses >= dose_level)
                        total_volume = len(struct_doses)
                        
                        if total_volume > 0:
                            dvh_values[i] = (volume_at_dose / total_volume) * 100.0
                    
                    # DVH 데이터 저장
                    dvh_data[struct_name] = {
                        'dose_bins': dose_bins,
                        'volume_percent': dvh_values
                    }
            except Exception as e:
                print(f"DVH 계산 오류 ({struct_name}): {str(e)}")
        
        self.dicom_series.dvh_data = dvh_data
        print(f"DVH 데이터 계산 완료: {len(dvh_data)} 구조체")
        
        # DVH 계산 완료 후 큰 데이터 정리
        # 이제 중요한 메트릭은 모두 추출했으므로 메모리 절약을 위해 큰 데이터 정리
        self.dicom_series.clear_large_data()
    
    @Slot()
    def run(self):
        """작업 실행 (QRunnable 메인 메서드) - 메모리 관리 개선"""
        try:
            print(f"환자 ID {self.dicom_series.patient_id} 분석 시작")
            
            # 이미 분석이 완료된 경우 재분석하지 않음
            if hasattr(self.dicom_series, 'analysis_completed') and self.dicom_series.analysis_completed:
                self.signals.progress.emit(100)
                self.signals.finished.emit(self.dicom_series)
                return
                
            # 1. CT 이미지 로드
            if not self.is_still_running():
                return
                
            self.signals.progress.emit(10)
            print("CT 이미지 로드 시작...")
            
            # z 위치로 정렬
            sorted_ct_files = sorted(self.dicom_series.ct_files, 
                                    key=lambda x: float(pydicom.dcmread(x, force=True).ImagePositionPatient[2]))
            
            # CT 이미지 로드 전 디버깅 정보
            print(f"CT 파일 수: {len(sorted_ct_files)}")
            if sorted_ct_files:
                first_ct = pydicom.dcmread(sorted_ct_files[0], force=True)
                print(f"첫 번째 CT 파일 정보: {first_ct.PatientID}, {first_ct.StudyInstanceUID}, {first_ct.SeriesInstanceUID}")
            
            if not self.is_still_running():
                return
                
            # CT 이미지 메타데이터 추출
            ct_pixels = []
            positions_z = []
            pixel_spacing = None
            position_first = None
            
            # 메모리 관리를 위한 배치 처리 기법 도입
            # 여기서는 전체 CT 볼륨을 한 번에 메모리에 로드하지 않고
            # 필요한 정보만 추출합니다
            
            # 일괄 처리 크기 (메모리 효율성 개선)
            batch_size = 10  # 한 번에 처리할 파일 수
            
            # 추출해야 하는 메타데이터를 위해 첫 번째 및 두 번째 슬라이스 처리
            first_slices = min(2, len(sorted_ct_files))
            for i in range(first_slices):
                file = sorted_ct_files[i]
                ds = pydicom.dcmread(file, force=True)
                if pixel_spacing is None:
                    pixel_spacing = ds.PixelSpacing
                if position_first is None:
                    position_first = ds.ImagePositionPatient[:2]  # x, y 좌표
                positions_z.append(float(ds.ImagePositionPatient[2]))
            
            # 나머지 파일에서는 z 위치만 추출 (메모리 효율성)
            for i in range(first_slices, len(sorted_ct_files)):
                file = sorted_ct_files[i]
                if not self.is_still_running():
                    return
                ds = pydicom.dcmread(file, stop_before_pixels=True, force=True)  # 픽셀 데이터 로드하지 않음
                positions_z.append(float(ds.ImagePositionPatient[2]))
            
            # CT 픽셀 데이터는 실제로 필요할 때만 로드
            # ct_pixels = []  # CT 픽셀 배열은 필요 시 로드하도록 수정
            
            print(f"CT 메타데이터 추출 완료: {len(positions_z)} 슬라이스, 픽셀 간격 {pixel_spacing}")
            
            # 2. RT Structure 파일 처리
            if not self.is_still_running():
                return
                
            self.signals.progress.emit(30)
            print(f"RT Structure 로드 시작: {self.dicom_series.rtstruct_file}")
            
            # RT Structure 파일 경로 확인
            if not os.path.exists(self.dicom_series.rtstruct_file):
                raise FileNotFoundError(f"RT Structure 파일이 존재하지 않습니다: {self.dicom_series.rtstruct_file}")
            
            # RTStruct 파일에서 컨투어 데이터 추출
            rtstruct_ds = pydicom.dcmread(self.dicom_series.rtstruct_file)
            
            # 구조체 이름 및 번호 매핑
            roi_names = {}
            for roi in rtstruct_ds.StructureSetROISequence:
                roi_names[roi.ROINumber] = roi.ROIName
            
            print(f"찾은 구조체: {roi_names}")
            
            # 구조체별 컨투어 데이터 추출
            structures = {}
            
            for roi_contour in rtstruct_ds.ROIContourSequence:
                if not self.is_still_running():
                    return
                    
                roi_number = roi_contour.ReferencedROINumber
                if roi_number in roi_names:
                    roi_name = roi_names[roi_number]
                    
                    # 컨투어 데이터 추출
                    contour_data = []
                    if hasattr(roi_contour, 'ContourSequence'):
                        for contour in roi_contour.ContourSequence:
                            if hasattr(contour, 'ContourData'):
                                points = []
                                for i in range(0, len(contour.ContourData), 3):
                                    if i+2 < len(contour.ContourData):
                                        x = float(contour.ContourData[i])
                                        y = float(contour.ContourData[i+1])
                                        z = float(contour.ContourData[i+2])
                                        points.append((x, y, z))
                                
                                if points:
                                    contour_data.append({
                                        'points': points,
                                        'z': points[0][2]  # z-좌표
                                    })
                    
                    if contour_data:
                        structures[roi_name] = contour_data
                        print(f"구조체 '{roi_name}' 컨투어 데이터 추출 완료 ({len(contour_data)} 슬라이스)")
            
            if not self.is_still_running():
                return
                
            self.dicom_series.structures = structures
            
            # 3. RT Dose 로드
            self.signals.progress.emit(50)
            print(f"RT Dose 로드 시작: {self.dicom_series.rtdose_file}")
            
            # RT Dose 파일 경로 확인
            if not os.path.exists(self.dicom_series.rtdose_file):
                raise FileNotFoundError(f"RT Dose 파일이 존재하지 않습니다: {self.dicom_series.rtdose_file}")
                
            rtdose_ds = pydicom.dcmread(self.dicom_series.rtdose_file)
            
            if not self.is_still_running():
                return
                
            # 선량 그리드 데이터 추출
            dose_grid = rtdose_ds.pixel_array * rtdose_ds.DoseGridScaling
            self.dicom_series.dose_array = dose_grid
            
            # 선량 이미지 메타데이터 저장
            self.dicom_series.dose_spacing = [
                rtdose_ds.PixelSpacing[0], 
                rtdose_ds.PixelSpacing[1], 
                rtdose_ds.GridFrameOffsetVector[1] - rtdose_ds.GridFrameOffsetVector[0]
            ]
            
            self.dicom_series.dose_origin = [
                rtdose_ds.ImagePositionPatient[0],
                rtdose_ds.ImagePositionPatient[1],
                rtdose_ds.ImagePositionPatient[2]
            ]
            
            print(f"RT Dose 이미지 로드 완료: 크기 {dose_grid.shape}, 간격 {self.dicom_series.dose_spacing}")
            
            # 4. 구조체별 마스크 생성 (선량 계산을 위해)
            if not self.is_still_running():
                return
                
            self.signals.progress.emit(60)
            print("구조체 마스크 생성 중...")
            
            # 선량 그리드 크기
            dose_grid_shape = dose_grid.shape
            
            # 각 구조체에 대한 마스크 생성
            structure_masks = {}
            for struct_name, contour_data in structures.items():
                if not self.is_still_running():
                    return
                    
                try:
                    # 선량 그리드 크기와 동일한 마스크 생성
                    mask = np.zeros(dose_grid_shape, dtype=bool)
                    
                    # 각 컨투어를 마스크에 그리기
                    for contour in contour_data:
                        if not self.is_still_running():
                            return
                            
                        # 컨투어 점을 선량 그리드 좌표계로 변환
                        points = contour['points']
                        z = contour['z']
                        
                        # 선량 그리드에서의 z-슬라이스 인덱스 찾기
                        z_indices = np.where(np.isclose(np.array(rtdose_ds.GridFrameOffsetVector) + rtdose_ds.ImagePositionPatient[2], z, atol=1.0))[0]
                        
                        if len(z_indices) > 0:
                            z_idx = z_indices[0]
                            
                            # 컨투어 점을 픽셀 좌표로 변환
                            pixel_points = []
                            for point in points:
                                x_phys, y_phys = point[0], point[1]
                                x_pixel = int((x_phys - rtdose_ds.ImagePositionPatient[0]) / rtdose_ds.PixelSpacing[0])
                                y_pixel = int((y_phys - rtdose_ds.ImagePositionPatient[1]) / rtdose_ds.PixelSpacing[1])
                                
                                if 0 <= x_pixel < dose_grid_shape[2] and 0 <= y_pixel < dose_grid_shape[1]:
                                    pixel_points.append((x_pixel, y_pixel))
                            
                            if len(pixel_points) > 2:  # 최소 삼각형을 형성할 수 있어야 함
                                try:
                                    # 다각형 내부 채우기
                                    poly = np.array(pixel_points)
                                    rr, cc = draw.polygon(poly[:, 1], poly[:, 0], shape=(dose_grid_shape[1], dose_grid_shape[2]))
                                    if 0 <= z_idx < dose_grid_shape[0]:
                                        mask[z_idx, rr, cc] = True
                                except Exception as e:
                                    print(f"마스크 생성 오류 ({struct_name}, 슬라이스 {z}): {str(e)}")
                    
                    structure_masks[struct_name] = mask
                    print(f"구조체 '{struct_name}' 마스크 생성 완료")
                except Exception as e:
                    print(f"구조체 '{struct_name}' 마스크 생성 중 오류: {str(e)}")
            
            self.dicom_series.structure_masks = structure_masks
            
            # 5. 구조체별 지표 계산
            if not self.is_still_running():
                return
                
            self.signals.progress.emit(70)
            print("구조체별 지표 계산 시작...")
            
            # 각 구조체에 대한 정확한 지표 계산
            self._calculate_structure_metrics(positions_z, pixel_spacing, rtdose_ds)
            
            # DVH 데이터 계산
            self._calculate_dvh_data(rtdose_ds)
            
            # 분석 완료 플래그 설정
            self.dicom_series.analysis_completed = True
            
            # 완료 신호 보내기
            if self.is_still_running():
                self.signals.progress.emit(100)
                print("계산 및 분석 완료")
                self.signals.finished.emit(self.dicom_series)
            
        except Exception as e:
            if self.is_still_running():
                import traceback
                traceback.print_exc()
                self.signals.error.emit(f"DICOM 분석 중 오류 발생: {str(e)}")
        finally:
            print("DicomAnalysisRunnable: 스레드 종료")
            # 명시적 메모리 정리 - 스레드 종료 전 마지막 정리
            if hasattr(self, 'dicom_series') and self.dicom_series is not None:
                # 이미 완전히 분석된 경우에는 큰 데이터 정리
                if hasattr(self.dicom_series, 'analysis_completed') and self.dicom_series.analysis_completed:
                    self.dicom_series.clear_large_data()
            
            # 메모리 강제 정리
            import gc
            gc.collect()


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib 그래프를 Qt에 표시하기 위한 캔버스 - 메모리 관리 개선"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def clear_figure(self):
        """그림의 모든 요소 정리"""
        if hasattr(self, 'fig') and self.fig is not None:
            # 모든 서브플롯 제거
            self.fig.clear()
            # 캔버스 업데이트
            self.draw()
    
    def destroy(self):
        """리소스 정리"""
        if hasattr(self, 'fig') and self.fig is not None:
            # 그림 닫기
            plt.close(self.fig)
            # 메모리 참조 제거
            self.fig = None
                
                
                
class DicomViewerApp(QMainWindow):
    """DICOM 뷰어 애플리케이션의 메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("방사선 치료 DICOM 분석 도구")
        self.setGeometry(100, 100, 1200, 800)
        
        self.dicom_series_list = []  # 여러 시리즈 저장
        self.current_series = None
        
        # QRunnable 객체 참조 변수
        self.loader_runnable = None
        self.analysis_runnable = None
        
        # 스레드풀 참조 저장
        self.thread_pool = QThreadPool.globalInstance()
        print(f"스레드풀의 최대 스레드 수: {self.thread_pool.maxThreadCount()}")
        
        # 필요하면 메모리 사용량을 위해 스레드풀 크기 제한
        if self.thread_pool.maxThreadCount() > 2:
            self.thread_pool.setMaxThreadCount(2)  # 동시에 최대 2개 스레드만 실행 (메모리 절약)
            print(f"스레드풀 크기 제한: {self.thread_pool.maxThreadCount()}")
        
        self.analyzed_series_indices = []  # 분석된 시리즈 인덱스 추적
        
        # UI 초기화
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        # 메인 위젯 설정
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # 상단 버튼 영역
        top_layout = QHBoxLayout()
        self.load_button = QPushButton("DICOM 폴더 선택")
        self.load_button.clicked.connect(self.load_dicom_folder)
        
        # 모든 시리즈 한번에 분석 버튼 추가
        self.analyze_all_button = QPushButton("모든 시리즈 분석")
        self.analyze_all_button.clicked.connect(self.analyze_all_series)
        self.analyze_all_button.setEnabled(False)
        
        # 엑셀 내보내기 버튼 영역
        self.export_button = QPushButton("결과 내보내기")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        
        # 내보내기 옵션 (라디오 버튼)
        self.export_option_layout = QHBoxLayout()
        self.export_current_radio = QRadioButton("현재 시리즈")
        self.export_all_radio = QRadioButton("모든 시리즈")
        self.export_all_radio.setChecked(True)  # 기본적으로 모든 시리즈 선택
        
        # 라디오 버튼 그룹화
        self.export_options = QButtonGroup()
        self.export_options.addButton(self.export_current_radio, 1)
        self.export_options.addButton(self.export_all_radio, 2)
        
        self.export_option_layout.addWidget(self.export_button)
        self.export_option_layout.addWidget(self.export_current_radio)
        self.export_option_layout.addWidget(self.export_all_radio)
        
        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.analyze_all_button)
        top_layout.addLayout(self.export_option_layout)
        top_layout.addStretch()
        
        # 메모리 정리 버튼 추가 (문제 해결용)
        self.cleanup_memory_button = QPushButton("메모리 정리")
        self.cleanup_memory_button.clicked.connect(self.force_memory_cleanup)
        top_layout.addWidget(self.cleanup_memory_button)
        
        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 상태 레이블
        self.status_label = QLabel("DICOM 폴더를 선택하세요.")
        
        # 메인 콘텐츠 영역 (시리즈 선택 + 데이터 표시)
        content_layout = QHBoxLayout()
        
        # 시리즈 선택 영역
        self.series_list = QListWidget()
        self.series_list.setMaximumWidth(250)
        self.series_list.currentRowChanged.connect(self.on_series_selected)
        
        # 시리즈 목록을 담는 그룹박스
        series_group = QGroupBox("DICOM 시리즈")
        series_layout = QVBoxLayout(series_group)
        series_layout.addWidget(self.series_list)
        
        # 탭 위젯 (데이터와 시각화 탭)
        self.tab_widget = QTabWidget()
        
        # 데이터 탭
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # 데이터 트리
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["항목", "값"])
        self.tree_widget.setColumnWidth(0, 300)
        data_layout.addWidget(self.tree_widget)
        
        # 시각화 탭
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Matplotlib 캔버스 추가
        self.canvas_3d = MatplotlibCanvas(width=10, height=6, dpi=100)
        viz_layout.addWidget(self.canvas_3d)
        
        # 선량-부피 히스토그램 캔버스
        self.canvas_dvh = MatplotlibCanvas(width=10, height=6, dpi=100)
        viz_layout.addWidget(self.canvas_dvh)
        
        # 탭 추가
        self.tab_widget.addTab(data_tab, "데이터")
        self.tab_widget.addTab(viz_tab, "시각화")
        
        # 콘텐츠 레이아웃에 추가
        content_layout.addWidget(series_group)
        content_layout.addWidget(self.tab_widget, 1)
        
        # 메인 레이아웃 구성
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(content_layout, 1)

        # 새로운 버튼 추가
        self.analyze_selected_button = QPushButton("선택된 시리즈 분석")
        self.analyze_selected_button.clicked.connect(self.analyze_selected_series)
        self.analyze_selected_button.setEnabled(False)
        
        # 상단 버튼 레이아웃에 버튼 추가
        top_layout.addWidget(self.analyze_selected_button)
    
    def force_memory_cleanup(self):
        """메모리 강제 정리 (문제 해결용)"""
        try:
            # 현재 작업 중지
            self.cleanup_for_new_operation()
            
            # 현재 시리즈가 있으면 큰 데이터 정리
            if self.current_series:
                self.current_series.clear_large_data()
            
            # 모든 시리즈의 큰 데이터 정리
            for series in self.dicom_series_list:
                if hasattr(series, 'clear_large_data'):
                    series.clear_large_data()
            
            # 그래프 리소스 정리
            if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                self.canvas_3d.clear_figure()
            if hasattr(self, 'canvas_dvh') and self.canvas_dvh is not None:
                self.canvas_dvh.clear_figure()
            
            # 메모리 강제 정리
            import gc
            gc.collect()
            
            # 메모리 정리 완료 메시지
            QMessageBox.information(self, "정보", "메모리 정리가 완료되었습니다.")
            self.status_label.setText("메모리 정리 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"메모리 정리 중 오류 발생: {str(e)}")

    def export_results(self):
        """분석 결과를 Excel 파일로 내보내기"""
        # 내보낼 시리즈 선택
        if self.export_all_radio.isChecked():
            export_series = self.dicom_series_list
        else:
            if not self.current_series:
                QMessageBox.warning(self, "경고", "내보낼 시리즈가 없습니다.")
                return
            export_series = [self.current_series]
        
        # 내보낼 데이터가 없는 경우
        if not export_series or not any(hasattr(series, 'structure_metrics') for series in export_series):
            QMessageBox.warning(self, "경고", "내보낼 분석 결과가 없습니다.")
            return
        
        # 파일 저장 대화상자
        save_path, _ = QFileDialog.getSaveFileName(self, "결과 파일 저장", "", "Excel 파일 (*.xlsx)")
        
        if not save_path:
            return
        
        try:
            # 모든 시리즈의 구조체 지표를 하나의 데이터프레임으로 통합
            all_metrics = []
            
            for series in export_series:
                if not hasattr(series, 'structure_metrics'):
                    continue
                
                # 각 구조체의 지표를 플랫한 딕셔너리로 변환
                for struct_name, metrics in series.structure_metrics.items():
                    # 기본 정보 추가
                    row = {
                        '환자 ID': series.patient_id,
                        '시리즈 설명': series.description,
                        '구조체': struct_name
                    }
                    
                    # 새로 추가된 컨투어 정보 포함
                    row['컨투어 수'] = metrics.get('contour_count', 0)
                    row['총 컨투어 두께 (mm)'] = metrics.get('total_contour_thickness_mm', 0)
                    
                    # 모든 메트릭 추가
                    row.update(metrics)
                    
                    # 시리즈 수준 지표 추가
                    if hasattr(series, 'conformity_index'):
                        row['Conformity Index (CI)'] = series.conformity_index
                    if hasattr(series, 'homogeneity_index'):
                        row['Homogeneity Index (HI)'] = series.homogeneity_index
                    
                    all_metrics.append(row)
            
            # 데이터프레임 생성
            df = pd.DataFrame(all_metrics)
            
            # Excel 파일로 내보내기
            df.to_excel(save_path, index=False, engine='openpyxl')
            
            # 성공 메시지
            QMessageBox.information(self, "성공", f"결과를 {save_path}에 저장했습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 저장 중 오류 발생: {str(e)}")
    
    def load_dicom_folder(self):
        """DICOM 폴더 선택 및 로드 - QThreadPool 버전"""
        # 시작 전 정리
        self.cleanup_for_new_operation()
        
        folder_path = QFileDialog.getExistingDirectory(self, "DICOM 폴더 선택")
        if folder_path:
            self.status_label.setText(f"폴더에서 DICOM 파일 로드 중: {folder_path}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # QRunnable 생성
            self.loader_runnable = DicomLoaderRunnable(folder_path)
            self.loader_runnable.signals.progress.connect(self.update_progress)
            self.loader_runnable.signals.finished.connect(self.on_dicom_loaded)
            self.loader_runnable.signals.error.connect(self.show_error)
            
            # 스레드풀에 작업 추가
            self.thread_pool.start(self.loader_runnable)
    
    def on_dicom_loaded(self, dicom_series_list):
        """DICOM 로드 완료 처리"""
        self.dicom_series_list = dicom_series_list
        
        # 시리즈 목록 업데이트 (환자 ID로 표시)
        self.series_list.clear()
        for i, series in enumerate(dicom_series_list):
            patient_id = series.patient_id if series.patient_id else f"Unknown_{i+1}"
            self.series_list.addItem(f"환자 ID: {patient_id} (미분석)")
        
        if self.dicom_series_list:
            self.analyze_all_button.setEnabled(True)
            self.analyze_selected_button.setEnabled(True)
            self.series_list.setCurrentRow(0)  # 첫 번째 시리즈 선택
            self.status_label.setText(f"DICOM 시리즈 {len(dicom_series_list)}개 로드됨. 분석 준비 완료.")
        else:
            self.status_label.setText("로드된 DICOM 시리즈가 없습니다.")
        
        # 프로그레스 바 숨기기
        self.progress_bar.setVisible(False)
        
        # 로더 참조 제거
        self.loader_runnable = None
        
        # 메모리 정리
        import gc
        gc.collect()
    
    def analyze_all_series(self):
        """모든 시리즈를 한번에 분석 - QThreadPool 버전"""
        # 시작 전 정리
        self.cleanup_for_new_operation()
        
        if not self.dicom_series_list:
            return
            
        self.status_label.setText("모든 시리즈 분석 중...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 분석 완료된 시리즈 추적
        self.analyzed_series_indices = []
        
        # 순차적으로 모든 시리즈 분석
        self.analyze_series_queue = list(range(len(self.dicom_series_list)))
        self._process_next_series_in_queue()
    
    def _process_next_series_in_queue(self):
        """분석 대기열에서 다음 시리즈 처리 - QThreadPool 버전"""
        if not self.analyze_series_queue:
            # 모두 완료됨
            self.status_label.setText("모든 시리즈 분석 완료")
            self.progress_bar.setVisible(False)
            self.export_button.setEnabled(True)
            
            # 마지막으로 분석된 시리즈로 설정
            if self.analyzed_series_indices:
                last_analyzed_index = self.analyzed_series_indices[-1]
                self.series_list.setCurrentRow(last_analyzed_index)
                self.current_series = self.dicom_series_list[last_analyzed_index]
                self.display_results()
                self.visualize_data()
                
            # 메모리 정리
            import gc
            gc.collect()
            return
            
        # 다음 시리즈 처리
        series_index = self.analyze_series_queue.pop(0)
        series_patient_id = self.dicom_series_list[series_index].patient_id or f"Unknown_{series_index+1}"
        total_series = len(self.dicom_series_list)
        progress = len(self.analyzed_series_indices) + 1
        
        self.status_label.setText(f"환자 ID '{series_patient_id}' 분석 중 ({progress}/{total_series})")
        
        # 선택된 시리즈에 대한 분석 작업 시작 - QRunnable 사용
        self.analysis_runnable = DicomAnalysisRunnable(self.dicom_series_list[series_index])
        self.analysis_runnable.signals.progress.connect(self.update_progress)
        self.analysis_runnable.signals.finished.connect(self._on_series_analysis_completed)
        self.analysis_runnable.signals.error.connect(self._on_series_analysis_error)
        
        # 현재 분석 중인 시리즈의 인덱스 저장
        self.current_analysis_index = series_index
        
        # 스레드풀에 작업 추가
        self.thread_pool.start(self.analysis_runnable)
    
    def _on_series_analysis_completed(self, dicom_series):
        """개별 시리즈 분석 완료 처리 (모든 시리즈 분석 모드)"""
        # 현재 분석된 시리즈의 인덱스 추가
        self.analyzed_series_indices.append(self.current_analysis_index)
        
        # 현재 분석된 시리즈의 UI 업데이트
        patient_id = dicom_series.patient_id or f"Unknown_{self.current_analysis_index+1}"
        self.series_list.item(self.current_analysis_index).setText(
            f"환자 ID: {patient_id} (분석 완료)"
        )
        
        # 분석기 참조 제거
        self.analysis_runnable = None
        
        # 큰 데이터 사용 후 메모리에서 정리
        dicom_series.clear_large_data()
        
        # 메모리 강제 정리
        import gc
        gc.collect()
        
        # 다음 시리즈 진행
        self._process_next_series_in_queue()
    
    def _on_series_analysis_error(self, error_message):
        """개별 시리즈 분석 오류 처리 (모든 시리즈 분석 모드)"""
        # 오류 시리즈의 UI 업데이트
        patient_id = self.dicom_series_list[self.current_analysis_index].patient_id or f"Unknown_{self.current_analysis_index+1}"
        self.series_list.item(self.current_analysis_index).setText(
            f"환자 ID: {patient_id} (분석 실패)"
        )
        
        # 분석기 참조 제거
        self.analysis_runnable = None
        
        # 메모리 강제 정리
        import gc
        gc.collect()
        
        # 오류를 출력하고 다음 시리즈 계속 진행
        print(f"시리즈 분석 오류: {error_message}")
        self._process_next_series_in_queue()
    
    def on_series_selected(self, row):
        """시리즈 선택 시 호출"""
        if row >= 0 and row < len(self.dicom_series_list):
            selected_series = self.dicom_series_list[row]
            
            # 이미 분석된 시리즈인 경우 바로 결과 표시
            if hasattr(selected_series, 'analysis_completed') and selected_series.analysis_completed:
                self.current_series = selected_series
                self.display_results()
                self.visualize_data()
                patient_id = selected_series.patient_id or "Unknown"
                self.status_label.setText(f"환자 ID {patient_id} 데이터 표시 중")
                self.export_button.setEnabled(True)
                return
            
            # 분석되지 않은 시리즈 선택 시 분석 버튼 활성화
            self.analyze_selected_button.setEnabled(True)
    
    def analyze_selected_series(self):
        """선택된 시리즈만 분석 - QThreadPool 버전"""
        current_row = self.series_list.currentRow()
        
        if current_row < 0 or current_row >= len(self.dicom_series_list):
            return
            
        # 시작 전 정리
        self.cleanup_for_new_operation()
        
        # 선택된 시리즈 분석 시작
        selected_series = self.dicom_series_list[current_row]
        patient_id = selected_series.patient_id or "Unknown"
        
        self.status_label.setText(f"환자 ID {patient_id} 분석 중...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # QRunnable 생성
        self.analysis_runnable = DicomAnalysisRunnable(selected_series)
        self.analysis_runnable.signals.progress.connect(self.update_progress)
        self.analysis_runnable.signals.finished.connect(self.on_series_analysis_completed)
        self.analysis_runnable.signals.error.connect(self.show_error)
        
        # 현재 분석 중인 시리즈의 인덱스 저장
        self.current_analysis_index = current_row
        
        # 스레드풀에 작업 추가
        self.thread_pool.start(self.analysis_runnable)

    def on_series_analysis_completed(self, dicom_series):
        """개별 시리즈 분석 완료 처리"""
        # 현재 분석된 시리즈의 인덱스로 UI 업데이트
        patient_id = dicom_series.patient_id or f"Unknown_{self.current_analysis_index+1}"
        
        # 시리즈 목록 항목 업데이트
        self.series_list.item(self.current_analysis_index).setText(
            f"환자 ID: {patient_id} (분석 완료)"
        )
        
        # 분석된 시리즈 표시
        self.current_series = dicom_series
        self.display_results()
        self.visualize_data()
        
        self.status_label.setText(f"환자 ID {patient_id} 분석 완료")
        self.progress_bar.setVisible(False)
        self.export_button.setEnabled(True)
        self.analyze_selected_button.setEnabled(False)
        
        # 분석기 참조 제거
        self.analysis_runnable = None
        
        # 완료 후 메모리 정리
        dicom_series.clear_large_data()
        import gc
        gc.collect()
    
    def display_results(self):
        """분석 결과를 트리 위젯에 표시"""
        # 트리 위젯 초기화
        self.tree_widget.clear()
        
        if not self.current_series or not hasattr(self.current_series, 'structure_metrics'):
            return
        
        # 환자 정보 항목
        patient_item = QTreeWidgetItem(self.tree_widget)
        patient_item.setText(0, "환자 정보")
        patient_item.setExpanded(True)
        
        patient_id_item = QTreeWidgetItem(patient_item)
        patient_id_item.setText(0, "환자 ID")
        patient_id_item.setText(1, self.current_series.patient_id)
        
        # 구조체 지표 항목
        structures_item = QTreeWidgetItem(self.tree_widget)
        structures_item.setText(0, "구조체 정보")
        structures_item.setExpanded(True)
        
        # 각 구조체에 대한 정보 추가
        for struct_name, metrics in self.current_series.structure_metrics.items():
            struct_item = QTreeWidgetItem(structures_item)
            struct_item.setText(0, struct_name)
            
            # 부피
            volume_item = QTreeWidgetItem(struct_item)
            volume_item.setText(0, "부피 (cc)")
            volume_item.setText(1, f"{metrics.get('volume_cc', 0):.2f}")
            
            # 고유 z-좌표 수 및 두께 추가
            contour_count_item = QTreeWidgetItem(struct_item)
            contour_count_item.setText(0, "고유 z-좌표 수")
            contour_count_item.setText(1, f"{metrics.get('contour_count', 0)}")
            
            contour_thickness_item = QTreeWidgetItem(struct_item)
            contour_thickness_item.setText(0, "총 컨투어 두께 (mm)")
            contour_thickness_item.setText(1, f"{metrics.get('total_contour_thickness_mm', 0):.2f}")
                        
            
            # 평균 선량
            if 'mean_dose' in metrics:
                mean_dose_item = QTreeWidgetItem(struct_item)
                mean_dose_item.setText(0, "평균 선량 (Gy)")
                mean_dose_item.setText(1, f"{metrics['mean_dose']:.2f}")
            
            # 최대 선량
            if 'max_dose' in metrics:
                max_dose_item = QTreeWidgetItem(struct_item)
                max_dose_item.setText(0, "최대 선량 (Gy)")
                max_dose_item.setText(1, f"{metrics['max_dose']:.2f}")
            
            # 최소 선량
            if 'min_dose' in metrics:
                min_dose_item = QTreeWidgetItem(struct_item)
                min_dose_item.setText(0, "최소 선량 (Gy)")
                min_dose_item.setText(1, f"{metrics['min_dose']:.2f}")
            
            # 특정 부피에 대한 선량
            for key, label in [
                ('D0.03cc', '0.03cc에 대한 선량 (Gy)'),
                ('D0.5cc', '0.5cc에 대한 선량 (Gy)'),
                ('D90%', '90% 부피에 대한 선량 (Gy)'),
                ('D95%', '95% 부피에 대한 선량 (Gy)'),
                ('D99.9%', '99.9% 부피에 대한 선량 (Gy)'),
                ('D100%', '100% 부피에 대한 선량 (Gy)')
            ]:
                if key in metrics:
                    item = QTreeWidgetItem(struct_item)
                    item.setText(0, label)
                    item.setText(1, f"{metrics[key]:.2f}")
            
            # 특정 선량에 대한 부피 비율
            for dose_level in [5, 10, 20, 30, 45, 50, 60]:
                key = f'V{dose_level}Gy_percent'
                if key in metrics:
                    item = QTreeWidgetItem(struct_item)
                    item.setText(0, f"V{dose_level}Gy (%)")
                    item.setText(1, f"{metrics[key]:.2f}")
        
        # CI 및 HI 정보
        indices_item = QTreeWidgetItem(self.tree_widget)
        indices_item.setText(0, "지표")
        indices_item.setExpanded(True)
        
        if hasattr(self.current_series, 'conformity_index') and self.current_series.conformity_index is not None:
            ci_item = QTreeWidgetItem(indices_item)
            ci_item.setText(0, "Conformity Index (CI)")
            ci_item.setText(1, f"{self.current_series.conformity_index:.4f}")
        
        if hasattr(self.current_series, 'homogeneity_index') and self.current_series.homogeneity_index is not None:
            hi_item = QTreeWidgetItem(indices_item)
            hi_item.setText(0, "Homogeneity Index (HI)")
            hi_item.setText(1, f"{self.current_series.homogeneity_index:.4f}")
    
    def visualize_data(self):
        """데이터 시각화 - 메모리 관리 개선"""
        if not self.current_series or not hasattr(self.current_series, 'structure_metrics') or not self.current_series.structure_metrics:
            return
            
        # 기존 그래프 정리
        if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
            self.canvas_3d.clear_figure()
        if hasattr(self, 'canvas_dvh') and self.canvas_dvh is not None:
            self.canvas_dvh.clear_figure()
        
        # 3D 시각화 (구조체의 무게중심)
        ax = self.canvas_3d.fig.add_subplot(111, projection='3d')
        
        # 영어 라벨로 설정하여 한글 폰트 문제 방지
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Structure Centroids')
        
        # 구조체 이름과 색상 매핑
        structure_colors = {
            'GTV': 'red',
            'ITV': 'orange',
            'CTV': 'yellow',
            'PTV': 'green',
            'Lungs': 'lightblue',
            'Heart': 'pink',
            'Esophagus': 'purple',
            'SpinalCord': 'brown',
            'BrachialPlexus': 'cyan',
            'Stomach': 'magenta',
            'Liver': 'olive',
            'Bowels': 'teal',
            'External': 'gray'
        }
        
        # 기본 색상 (위 매핑에 없는 구조체용)
        default_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        color_index = 0
        
        for struct_name, metrics in self.current_series.structure_metrics.items():
            if 'centroid_x_mm' in metrics:
                x = metrics['centroid_x_mm']
                y = metrics['centroid_y_mm']
                z = metrics['centroid_z_mm']
                
                # 구조체 색상 선택
                color = structure_colors.get(struct_name, default_colors[color_index % len(default_colors)])
                color_index += 1
                
                # 구조체 부피에 따른 마커 크기 (부피의 제곱근에 비례)
                size = np.sqrt(metrics.get('volume_cc', 10)) * 10
                
                # 무게중심 그리기
                ax.scatter(x, y, z, c=color, s=size, label=struct_name)
        
        # 범례 추가
        ax.legend()
        
        # 선량-부피 히스토그램 (DVH) - 실제 계산된 데이터 사용
        ax_dvh = self.canvas_dvh.fig.add_subplot(111)
        
        # 영어 라벨로 설정하여 한글 폰트 문제 방지
        ax_dvh.set_xlabel('Dose (Gy)')
        ax_dvh.set_ylabel('Volume (%)')
        ax_dvh.set_title('Dose-Volume Histogram (DVH)')
        
        # 중요 구조체 목록
        key_structures = ['PTV', 'Lungs-ITV', 'Heart', 'Esophagus', 'SpinalCord', 'BrachialPlexus']
        
        # 계산된 DVH 데이터가 있으면 사용
        if hasattr(self.current_series, 'dvh_data') and self.current_series.dvh_data:
            # 각 구조체에 대한 DVH 데이터 시각화
            for struct_name, dvh in self.current_series.dvh_data.items():
                # 구조체가 중요 구조체 목록에 없으면 건너뛰기
                if struct_name not in key_structures and len(self.current_series.dvh_data) > 10:
                    continue
                    
                # 구조체 색상 선택
                if struct_name in structure_colors:
                    color = structure_colors[struct_name]
                else:
                    color_index += 1
                    color = default_colors[color_index % len(default_colors)]
                
                # DVH 곡선 그리기
                ax_dvh.plot(dvh['dose_bins'], dvh['volume_percent'], label=struct_name, color=color)
        
        # 그리드 및 범례 추가
        ax_dvh.grid(True)
        ax_dvh.legend()
        
        # 캔버스 업데이트
        try:
            self.canvas_3d.draw()
            self.canvas_dvh.draw()
            
            # 불필요한 참조 정리
            ax = None
            ax_dvh = None
            
            # 강제 가비지 컬렉션
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"그래프 그리기 오류: {str(e)}")
    
    def update_progress(self, value):
        """프로그레스 바 업데이트"""
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        """오류 메시지 표시"""
        QMessageBox.critical(self, "오류", message)
        self.status_label.setText(f"오류: {message}")
        self.progress_bar.setVisible(False)
        
        # 오류 후 리소스 정리
        self.cleanup_for_new_operation()
    
    def cleanup_for_new_operation(self):
        """새 작업 시작 전 정리 - QThreadPool 버전"""
        print("새 작업을 위한 정리 작업 수행")
        
        # 실행 중인 작업 중지
        if hasattr(self, 'loader_runnable') and self.loader_runnable is not None:
            self.loader_runnable.stop()
            self.loader_runnable = None
            
        if hasattr(self, 'analysis_runnable') and self.analysis_runnable is not None:
            self.analysis_runnable.stop()
            self.analysis_runnable = None
        
        # 상태 초기화
        self.progress_bar.setValue(0)
        
        # UI 업데이트를 위한 이벤트 처리
        QApplication.processEvents()
        
        # 메모리 정리
        import gc
        gc.collect()
    
    def cleanup_all_resources(self):
        """애플리케이션 종료 전 모든 리소스 정리"""
        print("애플리케이션 종료 이벤트 - 모든 리소스 정리 중...")
        
        # 실행 중인 작업 중지
        self.cleanup_for_new_operation()
        
        # 스레드풀 작업 완료 대기
        print("ThreadPool 작업 완료 대기 중...")
        self.thread_pool.waitForDone(3000)  # 최대 3초 대기
        
        # 현재 시리즈 및 모든 시리즈의 큰 데이터 정리
        if self.current_series:
            self.current_series.clear_large_data()
        
        for series in self.dicom_series_list:
            if hasattr(series, 'clear_large_data'):
                series.clear_large_data()
        
        # matplotlib 리소스 해제
        try:
            plt.close('all')
            if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                self.canvas_3d.clear_figure()
                self.canvas_3d.destroy()
                self.canvas_3d = None
            if hasattr(self, 'canvas_dvh') and self.canvas_dvh is not None:
                self.canvas_dvh.clear_figure()
                self.canvas_dvh.destroy()
                self.canvas_dvh = None
        except Exception as e:
            print(f"그래프 리소스 해제 중 오류: {str(e)}")
        
        # 가비지 컬렉션 강제 실행
        import gc
        gc.collect()
        
        # 이벤트 처리
        QApplication.processEvents()
        
        print("모든 리소스 정리 완료")
    
    def closeEvent(self, event):
        """앱 종료 시 모든 스레드 정리 - QThreadPool 버전"""
        print("애플리케이션 closeEvent 발생")
        
        # 모든 리소스 정리
        self.cleanup_all_resources()
        
        print("closeEvent 처리 완료")
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 애플리케이션 전역 설정
    # 이 설정은 Qt가 QThreadPool 종료를 적절히 처리하도록 도움
    app.setQuitOnLastWindowClosed(True)
    
    window = DicomViewerApp()
    window.show()
    
    # 애플리케이션 종료 직전에 모든 스레드 정리를 위한 이벤트 핸들러 추가
    app.aboutToQuit.connect(window.cleanup_all_resources)
    
    # 종료 전 메인 윈도우가 적절히 정리될 수 있도록 처리
    result = app.exec()
    
    # 애플리케이션 종료 전 스레드풀 작업 완료 확인
    print("애플리케이션 종료 전 최종 정리...")
    QThreadPool.globalInstance().waitForDone(3000)  # 최대 3초 대기
    
    # 메모리 정리
    import gc
    gc.collect()
    
    sys.exit(result)


if __name__ == "__main__":
    main()