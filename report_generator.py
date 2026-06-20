
import io
import numpy as np
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm
import matplotlib.pyplot as plt


class ReportGenerator:
    """PDF报告生成器"""

    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate_report(self, params: Dict, results: Dict,
                        figures: List[plt.Figure] = None,
                        output_path: str = None) -> bytes:
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
        wq_data = [
            ['参数', '数值', '单位'],
            ['BOD衰减系数(K1)', str(params.get('K1', 0.25)), '1/d'],
            ['复氧系数(K2)', str(params.get('K2', 0.5)), '1/d'],
            ['扩散系数(Dx)', str(params.get('Dx', 10.0)), 'm²/s'],
            ['饱和溶解氧', str(params.get('D_O_sat', 9.5)), 'mg/L'],
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

        doc.build(story)

        pdf_bytes = buffer.getvalue()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        return pdf_bytes

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
