<<<<<<< HEAD
import React from "react";

export default function Help() {
  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded shadow mt-8">
      <h2 className="text-2xl font-bold mb-4">Help & Support</h2>
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Frequently Asked Questions</h3>
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>How do I upload media for analysis?</strong> Use the Dashboard to upload images, videos, audio, or use your webcam.</li>
          <li><strong>What file types are supported?</strong> JPEG, PNG, MP4, WAV, and more. See Settings for details.</li>
          <li><strong>How do I interpret the results?</strong> The dashboard and reports provide clear labels and explanations for each scan.</li>
          <li><strong>Is my data private?</strong> Yes, your uploads and results are processed securely and not shared.</li>
        </ul>
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-2">Contact Support</h3>
        <p>If you need further assistance, email <a href="mailto:support@satyaai.com" className="text-blue-600 underline">support@satyaai.com</a> or use the in-app chat.</p>
      </div>
    </div>
=======
import { Helmet } from 'react-helmet';
import { HelpCircle, Star, MessageCircle, Book, ExternalLink } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';

export default function Help() {
  const { toast } = useToast();
  const [contactName, setContactName] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [contactMessage, setContactMessage] = useState("");

  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // In a full implementation, this would send the form data to a server
    toast({
      title: "Message Sent",
      description: "Thank you for your message. We'll get back to you shortly.",
    });
    
    // Reset form
    setContactName("");
    setContactEmail("");
    setContactMessage("");
  };

  return (
    <>
      <Helmet>
        <title>SatyaAI - Help & Support</title>
        <meta name="description" content="Get help and learn more about using SatyaAI for deepfake detection. Access FAQs, guides, and contact support for assistance." />
      </Helmet>
      
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Help & Support</h1>
        <p className="text-muted-foreground">
          Learn more about SatyaAI and get help with using our deepfake detection tools.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <HelpCircle className="h-5 w-5 text-primary" />
                Frequently Asked Questions
              </CardTitle>
              <CardDescription>
                Common questions about SatyaAI and deepfake detection.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger>What is a deepfake?</AccordionTrigger>
                  <AccordionContent>
                    Deepfakes are synthetic media where a person's likeness is replaced with someone else's using artificial intelligence. This technology can create convincing images, videos, or audio recordings that depict people saying and doing things they never did in reality.
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="item-2">
                  <AccordionTrigger>How does SatyaAI detect deepfakes?</AccordionTrigger>
                  <AccordionContent>
                    SatyaAI uses a combination of advanced machine learning techniques to analyze media for inconsistencies that humans might miss. For images and videos, we examine facial landmarks, lighting inconsistencies, and unnatural movements. For audio, we analyze voice patterns and spectral inconsistencies typical of AI-generated content.
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="item-3">
                  <AccordionTrigger>How accurate is SatyaAI in detecting deepfakes?</AccordionTrigger>
                  <AccordionContent>
                    SatyaAI achieves a 96% accuracy rate in detecting deepfakes across various media types. However, as deepfake technology continues to evolve, we constantly update our algorithms to maintain high detection rates. The confidence score provided with each analysis gives you an indication of the reliability of our detection.
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="item-4">
                  <AccordionTrigger>What media formats are supported?</AccordionTrigger>
                  <AccordionContent>
                    For images, we support JPEG, PNG, and GIF formats. For videos, we support MP4, WebM, and MOV formats. For audio analysis, we support MP3, WAV, and OGG formats. There are file size limitations: 10MB for images and audio files, and 50MB for video files.
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="item-5">
                  <AccordionTrigger>Is my uploaded content private and secure?</AccordionTrigger>
                  <AccordionContent>
                    Yes, privacy is our priority. All uploaded content is processed on our secure servers and is not shared with third parties. Media files are only stored temporarily during analysis and are automatically deleted afterward unless you explicitly choose to save the results to your history.
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="item-6">
                  <AccordionTrigger>Can SatyaAI analyze live webcam feeds?</AccordionTrigger>
                  <AccordionContent>
                    Yes, SatyaAI can analyze live webcam feeds for real-time deepfake detection. This is particularly useful for verifying identity during video calls or online meetings. The webcam feature processes data locally in your browser for privacy, with only analysis results being sent to our servers.
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
            <CardFooter>
              <Button variant="outline" className="flex items-center gap-2 w-full">
                <Book className="h-4 w-4" />
                View Complete Documentation
              </Button>
            </CardFooter>
          </Card>
          
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Star className="h-5 w-5 text-primary" />
                Getting Started Guide
              </CardTitle>
              <CardDescription>
                Learn how to get the most out of SatyaAI's deepfake detection tools.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="border rounded-md p-4">
                  <h3 className="text-lg font-medium mb-2">Uploading Media for Analysis</h3>
                  <p className="text-muted-foreground mb-3">
                    You can upload images, videos, or audio files for analysis using the Upload section.
                    Simply drag and drop files or use the browse button to select media from your device.
                  </p>
                  <ol className="list-decimal ml-5 space-y-2 text-muted-foreground">
                    <li>Navigate to the Scan page or use the quick access tiles on the Dashboard.</li>
                    <li>Select the appropriate media type tab (Image, Video, or Audio).</li>
                    <li>Upload your file(s) by dragging and dropping or using the browse button.</li>
                    <li>Click "Start Analysis" to begin the deepfake detection process.</li>
                    <li>Review the detailed results once analysis is complete.</li>
                  </ol>
                </div>
                
                <div className="border rounded-md p-4">
                  <h3 className="text-lg font-medium mb-2">Understanding Analysis Results</h3>
                  <p className="text-muted-foreground mb-3">
                    After analysis is complete, SatyaAI provides detailed results with confidence scores and visualization.
                  </p>
                  <ul className="list-disc ml-5 space-y-2 text-muted-foreground">
                    <li>The overall result indicates whether the media is likely authentic or manipulated.</li>
                    <li>Confidence score shows how certain the system is about its detection (higher is more confident).</li>
                    <li>Detection details break down different aspects of the analysis with individual confidence scores.</li>
                    <li>Visual aids help identify specific areas of potential manipulation.</li>
                    <li>You can export results as PDF reports or CSV data for further analysis.</li>
                  </ul>
                </div>
                
                <div className="flex justify-center mt-4">
                  <Button className="flex items-center gap-2">
                    <ExternalLink className="h-4 w-4" />
                    Watch Video Tutorial
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageCircle className="h-5 w-5 text-primary" />
                Contact Support
              </CardTitle>
              <CardDescription>
                Need additional help? Reach out to our support team.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleContactSubmit} className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="name" className="text-sm font-medium">
                    Name
                  </label>
                  <Input 
                    id="name" 
                    placeholder="Your name" 
                    value={contactName}
                    onChange={(e) => setContactName(e.target.value)}
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <label htmlFor="email" className="text-sm font-medium">
                    Email
                  </label>
                  <Input 
                    id="email" 
                    type="email" 
                    placeholder="your.email@example.com" 
                    value={contactEmail}
                    onChange={(e) => setContactEmail(e.target.value)}
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <label htmlFor="message" className="text-sm font-medium">
                    Message
                  </label>
                  <Textarea 
                    id="message" 
                    placeholder="Describe your issue or question" 
                    rows={5} 
                    value={contactMessage}
                    onChange={(e) => setContactMessage(e.target.value)}
                    required
                  />
                </div>
                
                <Button type="submit" className="w-full">
                  Send Message
                </Button>
              </form>
            </CardContent>
          </Card>
          
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Resources</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <a 
                  href="#" 
                  className="block p-3 border rounded-md hover:bg-muted transition-colors"
                >
                  <h3 className="font-medium mb-1">Understanding Deepfakes</h3>
                  <p className="text-sm text-muted-foreground">
                    Learn about the technology behind deepfakes and their implications.
                  </p>
                </a>
                
                <a 
                  href="#" 
                  className="block p-3 border rounded-md hover:bg-muted transition-colors"
                >
                  <h3 className="font-medium mb-1">Detection Techniques Guide</h3>
                  <p className="text-sm text-muted-foreground">
                    Deep dive into the science of deepfake detection.
                  </p>
                </a>
                
                <a 
                  href="#" 
                  className="block p-3 border rounded-md hover:bg-muted transition-colors"
                >
                  <h3 className="font-medium mb-1">API Documentation</h3>
                  <p className="text-sm text-muted-foreground">
                    Integrate SatyaAI detection into your own applications.
                  </p>
                </a>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
  );
}
