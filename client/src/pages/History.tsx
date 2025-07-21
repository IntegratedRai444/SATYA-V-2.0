import React from 'react';
import { useState } from "react";
import { Helmet } from 'react-helmet';
import { useLocation } from "wouter";
import { useNavigation } from "../hooks/useNavigation";
import { useQuery } from "@tanstack/react-query";
import { ScanResult } from "../lib/types";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { Skeleton } from "../components/ui/skeleton";
import { Search, Filter, Trash2, FileDown } from "lucide-react";
import AnalysisResults from "../components/results/AnalysisResults";

export default function History() {
  const [location] = useLocation();
  const { navigate } = useNavigation();
  const [searchQuery, setSearchQuery] = useState("");
  const [mediaTypeFilter, setMediaTypeFilter] = useState("all");
  const [resultFilter, setResultFilter] = useState("all");
  
  // Get scan ID from URL if provided
  const scanId = location.split("/").pop();
  const isViewingSingleScan = scanId && scanId !== "history";
  
  // Fetch all scan history
  const { data: scanHistory, isLoading } = useQuery<ScanResult[]>({
    queryKey: ['/api/scans'],
  });
  
  // Filter results based on search and filters
  const filteredResults = scanHistory
    ? scanHistory.filter(scan => {
        // Apply search filter
        const matchesSearch = searchQuery === "" || 
          scan.filename.toLowerCase().includes(searchQuery.toLowerCase());
        
        // Apply media type filter
        const matchesType = mediaTypeFilter === "all" || 
          scan.type === mediaTypeFilter;
        
        // Apply result filter
        const matchesResult = resultFilter === "all" || 
          scan.result === resultFilter;
        
        return matchesSearch && matchesType && matchesResult;
      })
    : [];

  // Handle row click
  const handleRowClick = (id: string) => {
    navigate(`/history/${id}`);
  };

  return (
    <>
      <Helmet>
        <title>SatyaAI - Scan History</title>
        <meta name="description" content="View your past media scan history and deepfake detection results. Filter, search, and review detailed analysis of previously scanned media." />
      </Helmet>
      
      {isViewingSingleScan ? (
        // Show individual scan results
        <div>
          <div className="mb-6 flex justify-between items-center">
            <h1 className="text-3xl font-bold">Scan Details</h1>
            <Button variant="outline" onClick={() => navigate("/history")}>
              Back to History
            </Button>
          </div>
          <AnalysisResults scanId={scanId} />
        </div>
      ) : (
        // Show scan history list
        <div>
          <div className="mb-6">
            <h1 className="text-3xl font-bold mb-2">Scan History</h1>
            <p className="text-muted-foreground">
              View, filter and export your previous media analysis results.
            </p>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Media Analysis History</CardTitle>
              <CardDescription>
                A complete record of all media you've scanned for deepfake detection.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Filters and Search */}
              <div className="flex flex-col sm:flex-row gap-4 mb-6">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search by filename..."
                    className="pl-10"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                
                <div className="flex gap-2">
                  <Select value={mediaTypeFilter} onValueChange={setMediaTypeFilter}>
                    <SelectTrigger className="w-[150px]">
                      <div className="flex items-center gap-2">
                        <Filter className="h-4 w-4" />
                        <SelectValue placeholder="Media Type" />
                      </div>
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Types</SelectItem>
                      <SelectItem value="image">Images</SelectItem>
                      <SelectItem value="video">Videos</SelectItem>
                      <SelectItem value="audio">Audio</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  <Select value={resultFilter} onValueChange={setResultFilter}>
                    <SelectTrigger className="w-[150px]">
                      <div className="flex items-center gap-2">
                        <Filter className="h-4 w-4" />
                        <SelectValue placeholder="Result" />
                      </div>
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Results</SelectItem>
                      <SelectItem value="authentic">Authentic</SelectItem>
                      <SelectItem value="deepfake">Deepfake</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              {/* Results Table */}
              {isLoading ? (
                <HistorySkeleton />
              ) : filteredResults.length > 0 ? (
                <div className="rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Date & Time</TableHead>
                        <TableHead>Filename</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Result</TableHead>
                        <TableHead>Confidence</TableHead>
                        <TableHead className="w-[100px]">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredResults.map((scan) => (
                        <TableRow 
                          key={scan.id}
                          className="cursor-pointer"
                          onClick={() => handleRowClick(scan.id)}
                        >
                          <TableCell>{scan.timestamp}</TableCell>
                          <TableCell className="font-medium">{scan.filename}</TableCell>
                          <TableCell>
                            {scan.type.charAt(0).toUpperCase() + scan.type.slice(1)}
                          </TableCell>
                          <TableCell>
                            <span className={`px-2 py-1 rounded-full text-xs ${
                              scan.result === 'authentic' 
                                ? 'bg-accent/10 text-accent' 
                                : 'bg-destructive/10 text-destructive'
                            }`}>
                              {scan.result === 'authentic' ? 'Authentic' : 'Deepfake'}
                            </span>
                          </TableCell>
                          <TableCell>{scan.confidenceScore}%</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <FileDown className="h-4 w-4" />
                              </Button>
                              <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive">
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="border rounded-md p-12 text-center">
                  <h3 className="text-lg font-medium mb-2">No results found</h3>
                  <p className="text-muted-foreground mb-4">
                    {searchQuery || mediaTypeFilter !== "all" || resultFilter !== "all"
                      ? "Try adjusting your search or filters"
                      : "You haven't scanned any media yet"}
                  </p>
                  {searchQuery || mediaTypeFilter !== "all" || resultFilter !== "all" ? (
                    <Button onClick={() => {
                      setSearchQuery("");
                      setMediaTypeFilter("all");
                      setResultFilter("all");
                    }}>
                      Clear Filters
                    </Button>
                  ) : (
                    <Button onClick={() => navigate("/scan")}>
                      Scan New Media
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {filteredResults.length} of {scanHistory?.length || 0} results
              </div>
              <Button variant="outline" className="flex items-center gap-2">
                <FileDown className="h-4 w-4" />
                Export History
              </Button>
            </CardFooter>
          </Card>
        </div>
      )}
    </>
  );
}

function HistorySkeleton() {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date & Time</TableHead>
            <TableHead>Filename</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Result</TableHead>
            <TableHead>Confidence</TableHead>
            <TableHead className="w-[100px]">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Array(5).fill(0).map((_, i) => (
            <TableRow key={i}>
              <TableCell><Skeleton className="h-5 w-24" /></TableCell>
              <TableCell><Skeleton className="h-5 w-32" /></TableCell>
              <TableCell><Skeleton className="h-5 w-16" /></TableCell>
              <TableCell><Skeleton className="h-5 w-20" /></TableCell>
              <TableCell><Skeleton className="h-5 w-12" /></TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <Skeleton className="h-8 w-8 rounded" />
                  <Skeleton className="h-8 w-8 rounded" />
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
