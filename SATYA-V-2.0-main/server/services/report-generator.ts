/**
 * SatyaAI Report Generator Service
 * Generates detailed PDF reports from deepfake analysis results
 */
import fs from 'fs';
import path from 'path';
import { Response } from 'express';
import PDFDocument from 'pdfkit';
import { log } from '../vite';

interface ReportData {
  id: number;
  filename: string;
  type: string;
  result: string;
  confidence: number;
  analyzedAt: Date;
  details: {
    manipulationScore?: number;
    authenticityScore?: number;
    visualArtifacts?: number;
    audioArtifacts?: number;
    inconsistencyMarkers?: {
      facial?: number;
      audio?: number;
      metadata?: number;
    };
    detectedRegions?: {
      x: number;
      y: number;
      width: number;
      height: number;
      confidence: number;
      type: string;
    }[];
    metadata?: Record<string, any>;
    analysisTime?: number;
    modelVersion?: string;
    warnings?: string[];
    recommendations?: string[];
  };
}

/**
 * Generate a PDF report from analysis results
 */
export async function generatePDFReport(
  scanData: ReportData,
  res: Response
): Promise<void> {
  const doc = new PDFDocument({ 
    size: 'A4',
    margin: 50,
    info: {
      Title: `SatyaAI Analysis Report - ${scanData.filename}`,
      Author: 'SatyaAI Deepfake Detection System',
      Keywords: 'deepfake, analysis, media authentication',
      CreationDate: new Date()
    }
  });

  try {
    // Pipe the PDF to the response
    doc.pipe(res);
    
    // Add content to the PDF
    addReportHeader(doc, scanData);
    addSummarySection(doc, scanData);
    addDetailedAnalysis(doc, scanData);
    addTechnicalDetails(doc, scanData);
    addRecommendations(doc, scanData);
    addFooter(doc);
    
    // Finalize the PDF and end the stream
    doc.end();
    
    log(`Generated PDF report for scan ${scanData.id}`, 'report');
  } catch (error) {
    log(`Error generating PDF report: ${error}`, 'report');
    throw error;
  }
}

/**
 * Add header with logo and title
 */
function addReportHeader(doc: PDFKit.PDFDocument, data: ReportData): void {
  // Add logo
  // doc.image(path.join(process.cwd(), 'public', 'satyaai-logo.png'), 50, 45, { width: 50 });
  
  // Add report title
  doc.fontSize(20)
     .font('Helvetica-Bold')
     .fillColor('#00a8ff')
     .text('SatyaAI Deepfake Analysis Report', 50, 50)
     .moveDown(0.5);
  
  // Add report subtitle
  doc.fontSize(12)
     .font('Helvetica')
     .fillColor('#333333')
     .text(`Analysis of: ${data.filename}`, { align: 'left' })
     .moveDown(0.2);
  
  // Add date
  doc.fontSize(10)
     .fillColor('#666666')
     .text(`Generated on: ${new Date().toLocaleString()}`, { align: 'left' })
     .moveDown(0.5);
  
  // Add decorative line
  doc.strokeColor('#00a8ff')
     .lineWidth(1)
     .moveTo(50, 125)
     .lineTo(550, 125)
     .stroke()
     .moveDown(1);
}

/**
 * Add summary section with key findings
 */
function addSummarySection(doc: PDFKit.PDFDocument, data: ReportData): void {
  // Summary header
  doc.fontSize(16)
     .font('Helvetica-Bold')
     .fillColor('#00a8ff')
     .text('Summary', 50, 150)
     .moveDown(0.5);
  
  // Create summary box
  doc.roundedRect(50, 175, 500, 120, 5)
     .fillAndStroke('#f5f9ff', '#e0f0ff')
     .moveDown(0.5);
  
  // Result icon based on analysis result
  const resultIcon = data.result === 'AUTHENTIC' ? '✓' : '⚠';
  const resultColor = data.result === 'AUTHENTIC' ? '#209b5a' : '#d9534f';
  
  // Add result status
  doc.fontSize(14)
     .fillColor(resultColor)
     .text(resultIcon, 60, 185)
     .moveUp(1);
  
  doc.fontSize(14)
     .fillColor(resultColor)
     .text(`  ${data.result}`, { continued: true });
  
  // Add confidence score
  doc.fontSize(12)
     .fillColor('#666666')
     .text(`  (${data.confidence}% confidence)`, { align: 'left' })
     .moveDown(0.5);
  
  // Media information
  doc.fontSize(12)
     .fillColor('#333333')
     .text(`Media Type: ${data.type.toUpperCase()}`, 60, 220)
     .moveDown(0.3);
  
  doc.fontSize(12)
     .fillColor('#333333')
     .text(`Analysis Performed: ${new Date(data.analyzedAt).toLocaleString()}`, 60)
     .moveDown(0.3);
  
  // Analysis time if available
  if (data.details.analysisTime) {
    doc.fontSize(12)
       .fillColor('#333333')
       .text(`Processing Time: ${data.details.analysisTime.toFixed(2)} seconds`, 60)
       .moveDown(1);
  }
}

/**
 * Add detailed analysis section
 */
function addDetailedAnalysis(doc: PDFKit.PDFDocument, data: ReportData): void {
  // Section header
  doc.fontSize(16)
     .font('Helvetica-Bold')
     .fillColor('#00a8ff')
     .text('Detailed Analysis', 50, 320)
     .moveDown(0.5);
  
  // Create detailed analysis section
  doc.font('Helvetica')
     .fontSize(12)
     .fillColor('#333333');
  
  // Authentication scores
  if (data.details.authenticityScore !== undefined) {
    addScoreBar(doc, 'Authenticity Score', data.details.authenticityScore, 50, 350);
  }
  
  if (data.details.manipulationScore !== undefined) {
    addScoreBar(doc, 'Manipulation Probability', data.details.manipulationScore, 50, 390);
  }
  
  // Visual and audio artifacts if available
  let yPosition = 430;
  
  if (data.details.visualArtifacts !== undefined) {
    addScoreBar(doc, 'Visual Artifacts', data.details.visualArtifacts, 50, yPosition);
    yPosition += 40;
  }
  
  if (data.details.audioArtifacts !== undefined) {
    addScoreBar(doc, 'Audio Artifacts', data.details.audioArtifacts, 50, yPosition);
    yPosition += 40;
  }
  
  // Inconsistency markers
  if (data.details.inconsistencyMarkers) {
    doc.fontSize(14)
       .font('Helvetica-Bold')
       .fillColor('#333333')
       .text('Inconsistency Analysis', 50, yPosition)
       .moveDown(0.5);
    
    yPosition += 30;
    
    const markers = data.details.inconsistencyMarkers;
    
    if (markers.facial !== undefined) {
      addScoreBar(doc, 'Facial Inconsistencies', markers.facial, 50, yPosition);
      yPosition += 40;
    }
    
    if (markers.audio !== undefined) {
      addScoreBar(doc, 'Audio Inconsistencies', markers.audio, 50, yPosition);
      yPosition += 40;
    }
    
    if (markers.metadata !== undefined) {
      addScoreBar(doc, 'Metadata Inconsistencies', markers.metadata, 50, yPosition);
      yPosition += 40;
    }
  }
}

/**
 * Add technical details section
 */
function addTechnicalDetails(doc: PDFKit.PDFDocument, data: ReportData): void {
  // Make sure we have a new page for technical details
  doc.addPage();
  
  // Section header
  doc.fontSize(16)
     .font('Helvetica-Bold')
     .fillColor('#00a8ff')
     .text('Technical Details', 50, 50)
     .moveDown(0.5);
  
  // Model information
  doc.fontSize(12)
     .font('Helvetica-Bold')
     .fillColor('#333333')
     .text('Detection Models Used', 50)
     .moveDown(0.3);
  
  doc.fontSize(10)
     .font('Helvetica')
     .fillColor('#666666')
     .text(`Model Version: ${data.details.modelVersion || 'SatyaAI Neural Vision v4.2'}`)
     .moveDown(0.3);
  
  // Metadata information if available
  if (data.details.metadata && Object.keys(data.details.metadata).length > 0) {
    doc.fontSize(12)
       .font('Helvetica-Bold')
       .fillColor('#333333')
       .text('Extracted Metadata', 50)
       .moveDown(0.3);
    
    doc.fontSize(10)
       .font('Helvetica')
       .fillColor('#666666');
    
    for (const [key, value] of Object.entries(data.details.metadata)) {
      doc.text(`${key}: ${value}`)
         .moveDown(0.1);
    }
    
    doc.moveDown(0.5);
  }
  
  // Detection regions if available
  if (data.details.detectedRegions && data.details.detectedRegions.length > 0) {
    doc.fontSize(12)
       .font('Helvetica-Bold')
       .fillColor('#333333')
       .text('Detected Manipulation Regions', 50)
       .moveDown(0.3);
    
    doc.fontSize(10)
       .font('Helvetica')
       .fillColor('#666666');
    
    data.details.detectedRegions.forEach((region, index) => {
      doc.text(`Region #${index + 1}:`)
         .text(`  Type: ${region.type}`)
         .text(`  Confidence: ${region.confidence.toFixed(2)}%`)
         .text(`  Coordinates: x=${region.x}, y=${region.y}, width=${region.width}, height=${region.height}`)
         .moveDown(0.3);
    });
    
    doc.moveDown(0.5);
  }
}

/**
 * Add recommendations section
 */
function addRecommendations(doc: PDFKit.PDFDocument, data: ReportData): void {
  // Section header
  doc.fontSize(16)
     .font('Helvetica-Bold')
     .fillColor('#00a8ff')
     .text('Recommendations and Warnings', 50)
     .moveDown(0.5);
  
  // Warnings
  if (data.details.warnings && data.details.warnings.length > 0) {
    doc.fontSize(12)
       .font('Helvetica-Bold')
       .fillColor('#d9534f')
       .text('Warnings', 50)
       .moveDown(0.3);
    
    doc.fontSize(10)
       .font('Helvetica')
       .fillColor('#666666');
    
    data.details.warnings.forEach((warning, index) => {
      doc.text(`${index + 1}. ${warning}`)
         .moveDown(0.1);
    });
    
    doc.moveDown(0.5);
  } else {
    // Default warning based on result
    doc.fontSize(12)
       .font('Helvetica-Bold')
       .fillColor(data.result === 'AUTHENTIC' ? '#209b5a' : '#d9534f')
       .text(data.result === 'AUTHENTIC' ? 'No Warnings' : 'Potential Manipulation Detected', 50)
       .moveDown(0.3);
    
    doc.fontSize(10)
       .font('Helvetica')
       .fillColor('#666666');
    
    if (data.result !== 'AUTHENTIC') {
      doc.text('This media has been flagged as potentially manipulated. Review carefully before use.')
         .moveDown(0.5);
    } else {
      doc.text('No manipulation detected in this media. Always verify with multiple sources for critical use cases.')
         .moveDown(0.5);
    }
  }
  
  // Recommendations
  if (data.details.recommendations && data.details.recommendations.length > 0) {
    doc.fontSize(12)
       .font('Helvetica-Bold')
       .fillColor('#00a8ff')
       .text('Recommendations', 50)
       .moveDown(0.3);
    
    doc.fontSize(10)
       .font('Helvetica')
       .fillColor('#666666');
    
    data.details.recommendations.forEach((recommendation, index) => {
      doc.text(`${index + 1}. ${recommendation}`)
         .moveDown(0.1);
    });
  } else {
    // Default recommendations
    doc.fontSize(12)
       .font('Helvetica-Bold')
       .fillColor('#00a8ff')
       .text('General Recommendations', 50)
       .moveDown(0.3);
    
    doc.fontSize(10)
       .font('Helvetica')
       .fillColor('#666666')
       .text('1. Always verify important media with multiple detection tools.')
       .moveDown(0.1)
       .text('2. Check the source of the media and its chain of custody.')
       .moveDown(0.1)
       .text('3. For critical decisions, consult with a digital forensics expert.')
       .moveDown(0.1)
       .text('4. Keep your deepfake detection tools updated as technology evolves.')
       .moveDown(0.5);
  }
}

/**
 * Add footer with disclaimer and contact info
 */
function addFooter(doc: PDFKit.PDFDocument): void {
  // Add decorative line
  doc.strokeColor('#00a8ff')
     .lineWidth(1)
     .moveTo(50, 700)
     .lineTo(550, 700)
     .stroke()
     .moveDown(1);
  
  // Add disclaimer
  doc.fontSize(8)
     .font('Helvetica-Oblique')
     .fillColor('#999999')
     .text('DISCLAIMER: This report is generated using advanced AI technology, but should not be considered as definitive proof of authenticity or manipulation. Results may vary based on media quality and manipulation techniques. For legal or critical applications, please consult with digital forensics experts.', 50, 710, { align: 'center', width: 500 })
     .moveDown(0.5);
  
  // Add contact information
  doc.fontSize(8)
     .font('Helvetica')
     .fillColor('#999999')
     .text('SatyaAI Deepfake Detection System • Generated with SatyaAI Analysis Tools • support@satyaai.com', { align: 'center' })
     .moveDown(0.2);
  
  // Add page number
  const pageCount = doc.bufferedPageRange().count;
  for (let i = 0; i < pageCount; i++) {
    doc.switchToPage(i);
    doc.fontSize(8)
       .fillColor('#999999')
       .text(
         `Page ${i + 1} of ${pageCount}`,
         50,
         doc.page.height - 50,
         { align: 'center', width: 500 }
       );
  }
}

/**
 * Helper function to create score bars
 */
function addScoreBar(
  doc: PDFKit.PDFDocument,
  label: string,
  score: number,
  x: number,
  y: number
): void {
  const barWidth = 400;
  const barHeight = 20;
  
  // Normalize score between 0-100
  const normalizedScore = Math.min(Math.max(score, 0), 100);
  
  // Label
  doc.fontSize(10)
     .font('Helvetica')
     .fillColor('#333333')
     .text(label, x, y);
  
  // Score value
  doc.fontSize(10)
     .font('Helvetica-Bold')
     .fillColor('#333333')
     .text(`${normalizedScore.toFixed(1)}%`, x + barWidth + 15, y);
  
  // Background bar
  doc.roundedRect(x, y + 15, barWidth, barHeight, 3)
     .fillAndStroke('#eeeeee', '#dddddd');
  
  // Calculate gradient color based on score type and value
  let fillColor;
  const isGoodHighScore = label.includes('Authenticity');
  const isGoodLowScore = label.includes('Manipulation') || label.includes('Artifacts') || label.includes('Inconsistencies');
  
  if (isGoodHighScore) {
    // Green for high authenticity
    fillColor = normalizedScore > 80 ? '#4caf50' : 
                normalizedScore > 50 ? '#ffc107' : '#f44336';
  } else if (isGoodLowScore) {
    // Red for high manipulation/artifacts
    fillColor = normalizedScore < 20 ? '#4caf50' : 
                normalizedScore < 50 ? '#ffc107' : '#f44336';
  } else {
    // Default blue gradient
    fillColor = '#00a8ff';
  }
  
  // Score fill bar
  doc.roundedRect(x, y + 15, (barWidth * normalizedScore) / 100, barHeight, 3)
     .fill(fillColor);
}

/**
 * Generate a PDF buffer instead of streaming to response
 * Useful for saving to disk or sending via email
 */
export async function generatePDFBuffer(scanData: ReportData): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const doc = new PDFDocument({ 
      size: 'A4',
      margin: 50,
      info: {
        Title: `SatyaAI Analysis Report - ${scanData.filename}`,
        Author: 'SatyaAI Deepfake Detection System',
        Keywords: 'deepfake, analysis, media authentication',
        CreationDate: new Date()
      }
    });

    doc.on('data', (chunk) => chunks.push(chunk));
    doc.on('end', () => resolve(Buffer.concat(chunks)));
    doc.on('error', reject);

    try {
      // Add content to the PDF
      addReportHeader(doc, scanData);
      addSummarySection(doc, scanData);
      addDetailedAnalysis(doc, scanData);
      addTechnicalDetails(doc, scanData);
      addRecommendations(doc, scanData);
      addFooter(doc);
      
      // Finalize the PDF
      doc.end();
    } catch (error) {
      reject(error);
    }
  });
}