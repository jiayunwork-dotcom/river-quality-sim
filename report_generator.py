
import io
import numpy as np
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether
)
from reportlab.lib.units import cm
import matplotlib.pyplot as plt


WATER_QUALITY_STANDARDS = {
    'bod': 4.0,
    'do': 5.0,
    'nh3n': 1.0,
}


class ReportGenerator:
    """PDF报告生成器"""

    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate_report(self, params: Dict, results: Dict,
                        figures: List[plt.Figure] = None,
                        output_path: str = None,
                        compliance_summary: Dict = None) -> bytes:
        """生成PDF报告"""
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        story = []

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#34495e')
        )

        normal_style = self.styles['Normal']

        story.append(Paragraph("河流水质模拟分析报告", title_style))
        story.append(Spacer(1, 0.5*cm))

        story.append(Paragraph("一、参数设置", heading_style))
        param_data = [
            ['参数名称', '数值', '单位'],
        ]

        channel_params = params.get('channel', {})
        for key, value in channel_params.items():
            param_data.append([key, str(value), ''])

        param_table = Table(param_data, colWidths=[5*cm, 5*cm, 3*cm])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ]))
        story.append(param_table)

        story.append(Paragraph("二、水质参数", heading_style))

        def _fmt(val, decimals=4):
            try:
                return f"{float(val):.{decimals}f}"
            except (ValueError, TypeError):
                return str(val)

        wq_data = [
            ['参数', '数值', '单位'],
            ['BOD衰减系数(K1)', _fmt(params.get('K1', 0.25)), '1/d'],
            ['复氧系数(K2)', _fmt(params.get('K2', 0.5)), '1/d'],
            ['氨氮衰减系数(K_nh3n)', _fmt(params.get('K_nh3n', 0.1)), '1/d'],
            ['COD衰减系数(K_cod)', _fmt(params.get('K_cod', 0.15)), '1/d'],
            ['扩散系数(Dx)', _fmt(params.get('Dx', 10.0), 2), 'm²/s'],
            ['饱和溶解氧(DO_sat)', _fmt(params.get('D_O_sat', 9.5), 2), 'mg/L'],
        ]
        wq_table = Table(wq_data, colWidths=[5*cm, 5*cm, 3*cm])
        wq_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ]))
        story.append(wq_table)

        story.append(Paragraph("三、计算结果汇总", heading_style))
        result_summary = [
            ['指标', '最大值', '最小值', '平均值', '单位'],
        ]

        components = {
            'BOD': 'bod',
            'DO': 'do',
            'NH3-N': 'nh3n',
            'COD': 'cod',
        }

        for name, key in components.items():
            data = results.get(key, np.array([0]))
            result_summary.append([
                name,
                f"{np.max(data):.3f}",
                f"{np.min(data):.3f}",
                f"{np.mean(data):.3f}",
                'mg/L'
            ])

        result_table = Table(result_summary, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 2*cm])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ]))
        story.append(result_table)

        if results.get('critical_x') is not None:
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph(
                f"临界点位置：距离起点 {results['critical_x']:.1f} m，"
                f"最低DO浓度：{results['critical_do']:.2f} mg/L",
                normal_style
            ))

        story.append(Paragraph("四、各断面详细结果", heading_style))

        x = results.get('x', np.array([]))
        if len(x) > 0:
            detail_data = [['距离(m)', '水深(m)', '流速(m/s)', 'BOD(mg/L)', 'DO(mg/L)', 'NH3-N(mg/L)', 'COD(mg/L)']]

            step = max(1, len(x) // 20)
            for i in range(0, len(x), step):
                detail_data.append([
                    f"{x[i]:.1f}",
                    f"{results.get('h', np.zeros_like(x))[i]:.3f}",
                    f"{results.get('V', np.zeros_like(x))[i]:.3f}",
                    f"{results.get('bod', np.zeros_like(x))[i]:.3f}",
                    f"{results.get('do', np.zeros_like(x))[i]:.3f}",
                    f"{results.get('nh3n', np.zeros_like(x))[i]:.3f}",
                    f"{results.get('cod', np.zeros_like(x))[i]:.3f}",
                ])

            detail_table = Table(detail_data, colWidths=[2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 2*cm, 1.8*cm])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ]))
            story.append(detail_table)

        if figures:
            story.append(Paragraph("五、图表", heading_style))
            for i, fig in enumerate(figures):
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img = Image(img_buffer, width=15*cm, height=10*cm)
                story.append(img)
                story.append(Spacer(1, 0.5*cm))

        if compliance_summary is None:
            compliance_summary = self._compute_compliance_summary(results)

        self._append_compliance_summary_page(story, heading_style, compliance_summary, normal_style)

        doc.build(story)

        pdf_bytes = buffer.getvalue()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        return pdf_bytes

    def _compute_compliance_at_point(self, bod_val, do_val, nh3n_val):
        bod_ok = bod_val <= WATER_QUALITY_STANDARDS['bod']
        do_ok = do_val >= WATER_QUALITY_STANDARDS['do']
        nh3n_ok = nh3n_val <= WATER_QUALITY_STANDARDS['nh3n']
        return bod_ok and do_ok and nh3n_ok

    def _compute_compliance_array(self, result):
        x = result['x']
        n = len(x)
        compliance = np.zeros(n, dtype=bool)
        for i in range(n):
            compliance[i] = self._compute_compliance_at_point(
                result['bod'][i], result['do'][i], result['nh3n'][i]
            )
        return compliance

    def _compute_sliding_compliance_rate(self, result, window=500.0):
        x = result['x']
        n = len(x)
        compliance = self._compute_compliance_array(result)
        rates = np.zeros(n)
        for i in range(n):
            x_center = x[i]
            mask = (x >= x_center - window) & (x <= x_center + window)
            window_points = compliance[mask]
            if len(window_points) > 0:
                rates[i] = np.sum(window_points) / len(window_points) * 100
            else:
                rates[i] = 100.0
        return x, rates

    def _compute_compliance_summary(self, result):
        """计算水质达标评估摘要（与app.py中一致的算法）"""
        x = result.get('x', np.array([]))
        summary = {
            'avg_compliance_rate': 100.0,
            'worst_position': 0.0,
            'worst_items': ['无'],
            'exceed_length_ratio': 0.0,
            'exceed_length': 0.0,
            'conclusion': '该河段水质整体满足地表水III类标准，水环境质量良好。',
            'most_severe_comp': '-',
            'most_severe_pct': 0.0,
            'start_seg': '-',
            'end_seg': '-',
        }
        if len(x) < 2:
            return summary

        total_length = x[-1] - x[0]
        _, rates = self._compute_sliding_compliance_rate(result)
        avg_rate = float(np.mean(rates))
        summary['avg_compliance_rate'] = avg_rate

        compliance = self._compute_compliance_array(result)
        non_compliance_mask = ~compliance

        worst_idx = None
        worst_score = float('inf')
        for i in range(len(x)):
            score = 0
            if result['bod'][i] > WATER_QUALITY_STANDARDS['bod']:
                score += (result['bod'][i] / WATER_QUALITY_STANDARDS['bod'] - 1) * 100
            if result['do'][i] < WATER_QUALITY_STANDARDS['do']:
                score += (WATER_QUALITY_STANDARDS['do'] / result['do'][i] - 1) * 100 if result['do'][i] > 0 else 1000
            if result['nh3n'][i] > WATER_QUALITY_STANDARDS['nh3n']:
                score += (result['nh3n'][i] / WATER_QUALITY_STANDARDS['nh3n'] - 1) * 100
            if score > 0 and score < worst_score:
                worst_score = score
                worst_idx = i

        worst_x = float(x[worst_idx]) if worst_idx is not None else float(x[0])
        worst_items = []
        if worst_idx is not None:
            if result['bod'][worst_idx] > WATER_QUALITY_STANDARDS['bod']:
                worst_items.append('BOD')
            if result['do'][worst_idx] < WATER_QUALITY_STANDARDS['do']:
                worst_items.append('DO')
            if result['nh3n'][worst_idx] > WATER_QUALITY_STANDARDS['nh3n']:
                worst_items.append('NH3-N')
        if not worst_items:
            worst_items = ['无']
        summary['worst_position'] = worst_x
        summary['worst_items'] = worst_items

        exceed_segments_length = 0.0
        if np.any(non_compliance_mask):
            dx = x[1] - x[0]
            exceed_segments_length = float(np.sum(non_compliance_mask) * dx)
        summary['exceed_length'] = exceed_segments_length
        summary['exceed_length_ratio'] = exceed_segments_length / total_length * 100 if total_length > 0 else 0

        bod_exceed_total = np.sum(result['bod'] > WATER_QUALITY_STANDARDS['bod'])
        do_below_total = np.sum(result['do'] < WATER_QUALITY_STANDARDS['do'])
        nh3n_exceed_total = np.sum(result['nh3n'] > WATER_QUALITY_STANDARDS['nh3n'])

        most_severe_comp = 'BOD'
        most_severe_pct = 0.0
        if bod_exceed_total > 0:
            max_excess = (np.max(result['bod']) / WATER_QUALITY_STANDARDS['bod'] - 1) * 100
            most_severe_pct = float(max_excess)
        if do_below_total > 0:
            min_do = np.min(result['do'])
            excess = (WATER_QUALITY_STANDARDS['do'] / min_do - 1) * 100 if min_do > 0 else 999
            if excess > most_severe_pct:
                most_severe_pct = float(excess)
                most_severe_comp = 'DO'
        if nh3n_exceed_total > 0:
            max_excess = (np.max(result['nh3n']) / WATER_QUALITY_STANDARDS['nh3n'] - 1) * 100
            if max_excess > most_severe_pct:
                most_severe_pct = float(max_excess)
                most_severe_comp = 'NH3-N'
        summary['most_severe_comp'] = most_severe_comp
        summary['most_severe_pct'] = most_severe_pct

        non_comp_indices = np.where(non_compliance_mask)[0]
        start_seg = end_seg = '-'
        if len(non_comp_indices) > 0:
            start_seg = f"{x[non_comp_indices[0]]:.0f}m"
            end_seg = f"{x[non_comp_indices[-1]]:.0f}m"
        summary['start_seg'] = start_seg
        summary['end_seg'] = end_seg

        if most_severe_pct == 0:
            conclusion = "该河段水质整体满足地表水III类标准，水环境质量良好。"
        else:
            conclusion = (f"该河段{most_severe_comp}超标严重（最大超标{most_severe_pct:.1f}%），"
                          f"建议重点治理{start_seg}至{end_seg}段。")
        summary['conclusion'] = conclusion

        return summary

    def _append_compliance_summary_page(self, story, heading_style, summary, normal_style):
        """在story末尾追加水质达标评估摘要页"""
        story.append(PageBreak())

        title_style = ParagraphStyle(
            'SummaryTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50')
        )

        story.append(Paragraph("六、水质达标评估摘要", title_style))
        story.append(Spacer(1, 0.3*cm))

        standards_style = ParagraphStyle(
            'StandardsText',
            parent=normal_style,
            textColor=colors.HexColor('#555555'),
            fontSize=10,
            spaceAfter=15,
            leftIndent=0,
        )
        story.append(Paragraph(
            "评估标准：地表水III类 — DO ≥ 5mg/L | BOD ≤ 4mg/L | NH3-N ≤ 1mg/L "
            "（达标率基于±500m滑动窗口内三项指标全部达标的网格点比例）",
            standards_style
        ))

        kpi_data = [
            ['评估指标', '结果'],
            ['全河段平均达标率', f"{summary.get('avg_compliance_rate', 100):.1f} %"],
            ['最差断面位置', f"{summary.get('worst_position', 0):.0f} m"],
            ['最差断面超标项目', '、'.join(summary.get('worst_items', ['无']))],
            ['超标河段总长度', f"{summary.get('exceed_length', 0):.0f} m"],
            ['超标河段长度占比', f"{summary.get('exceed_length_ratio', 0):.1f} %"],
        ]

        kpi_table = Table(kpi_data, colWidths=[7*cm, 8*cm])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16a085')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#ecf0f1')),
        ]))

        avg_rate = summary.get('avg_compliance_rate', 100)
        if avg_rate < 60:
            rate_bg = colors.HexColor('#fdecea')
            rate_text = colors.HexColor('#c0392b')
        elif avg_rate < 80:
            rate_bg = colors.HexColor('#fef5e7')
            rate_text = colors.HexColor('#d35400')
        else:
            rate_bg = colors.HexColor('#e8f8f5')
            rate_text = colors.HexColor('#138d75')
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (1, 1), (1, 1), rate_bg),
            ('TEXTCOLOR', (1, 1), (1, 1), rate_text),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ]))

        story.append(kpi_table)
        story.append(Spacer(1, 0.8*cm))

        conclusion_heading_style = ParagraphStyle(
            'ConclusionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        )
        story.append(Paragraph("评估结论", conclusion_heading_style))

        conclusion_style = ParagraphStyle(
            'ConclusionText',
            parent=normal_style,
            fontSize=12,
            leading=18,
            leftIndent=6,
            textColor=colors.HexColor('#2c3e50'),
            borderColor=colors.HexColor('#3498db'),
            borderWidth=1,
            borderPadding=10,
            backColor=colors.HexColor('#ebf5fb'),
        )
        story.append(Paragraph(summary.get('conclusion', ''), conclusion_style))
        story.append(Spacer(1, 0.8*cm))

        footer_style = ParagraphStyle(
            'FooterText',
            parent=normal_style,
            fontSize=9,
            textColor=colors.HexColor('#888888'),
            alignment=1,
        )
        story.append(Spacer(1, 1.0*cm))
        story.append(Paragraph(
            "—— 本评估基于稳态模拟结果，仅供参考 ——",
            footer_style
        ))

    def generate_comparison_report(self, scenarios: List[Dict],
                                    results: List[Dict],
                                    output_path: str = None) -> bytes:
        """生成多情景对比报告"""
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        story = []

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#2c3e50')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#34495e')
        )

        story.append(Paragraph("多情景对比分析报告", title_style))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("情景对比汇总表", heading_style))

        table_data = [['情景名称', '最大BOD(mg/L)', '最低DO(mg/L)', '最大NH3-N(mg/L)', '最大COD(mg/L)']]

        for i, (scenario, result) in enumerate(zip(scenarios, results)):
            table_data.append([
                scenario.get('name', f'情景{i+1}'),
                f"{np.max(result.get('bod', [0])):.3f}",
                f"{np.min(result.get('do', [0])):.3f}",
                f"{np.max(result.get('nh3n', [0])):.3f}",
                f"{np.max(result.get('cod', [0])):.3f}",
            ])

        comp_table = Table(table_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8e44ad')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f5f5f5')),
        ]))
        story.append(comp_table)

        doc.build(story)

        pdf_bytes = buffer.getvalue()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        return pdf_bytes
