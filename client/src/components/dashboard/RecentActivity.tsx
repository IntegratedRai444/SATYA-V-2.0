import { useState } from "react";
import { Image, Video, Mic } from "lucide-react";
import { cn } from "../../lib/utils";
import { Skeleton } from "../ui/skeleton";
import { Button } from "../ui/button";
import { useQuery } from "@tanstack/react-query";
import { ScanResult } from "../../lib/types";
import { useNavigation } from "../../hooks/useNavigation";

export default function RecentActivity() {
  const { navigate } = useNavigation();
  
  // Fetch recent activities
  const { data: recentActivities, isLoading } = useQuery<ScanResult[]>({
    queryKey: ['/api/scans/recent'],
  });

  const getIcon = (type: string) => {
    switch (type) {
      case 'image':
        return <Image className="text-accent" size={18} />;
      case 'video':
        return <Video className="text-destructive" size={18} />;
      case 'audio':
        return <Mic className="text-secondary" size={18} />;
      default:
        return <Image className="text-accent" size={18} />;
    }
  };

  const getBorderColor = (result: string) => {
    return result === 'authentic' ? 'border-accent' : 'border-destructive';
  };

  const getLabelColor = (result: string) => {
    if (result === 'authentic') {
      return 'text-accent bg-accent/10';
    } else {
      return 'text-destructive bg-destructive/10';
    }
  };

  const handleViewHistory = () => {
    navigate("/history");
  };

  return (
    <div className="bg-card rounded-xl p-6">
      <h2 className="text-xl font-poppins font-semibold text-foreground mb-4 flex items-center">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-primary mr-2"
        >
          <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z" />
          <path d="M12 7v5l3 3" />
        </svg>
        Recent Activity
      </h2>

      <div className="space-y-4">
        {isLoading ? (
          // Skeleton loading state
          Array(3)
            .fill(0)
            .map((_, index) => (
              <div key={index} className="p-3 rounded-lg bg-muted">
                <div className="flex items-start">
                  <Skeleton className="h-10 w-10 rounded-full mr-3" />
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <Skeleton className="h-4 w-24" />
                      <Skeleton className="h-4 w-16" />
                    </div>
                    <Skeleton className="h-3 w-32 mt-2" />
                    <Skeleton className="h-3 w-24 mt-2" />
                  </div>
                </div>
              </div>
            ))
        ) : (
          <>
            {recentActivities?.length ? (
              recentActivities.map((activity) => (
                <div
                  key={activity.id}
                  className="p-3 rounded-lg bg-muted hover:bg-muted/70 transition-colors cursor-pointer"
                  onClick={() => navigate(`/history/${activity.id}`)}
                >
                  <div className="flex items-start">
                    <div className="mr-3 mt-1">
                      <div className={cn(
                        "w-10 h-10 rounded-full bg-card flex items-center justify-center border",
                        getBorderColor(activity.result)
                      )}>
                        {getIcon(activity.type)}
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between items-start">
                        <h4 className="font-medium text-foreground">{activity.filename}</h4>
                        <span className={cn(
                          "text-xs px-2 py-1 rounded-full",
                          getLabelColor(activity.result)
                        )}>
                          {activity.result === 'authentic' ? 'Authentic' : 'Deepfake'}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {activity.confidenceScore}% confidence score
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {activity.timestamp}
                      </p>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center p-6">
                <p className="text-muted-foreground mb-4">No recent activities</p>
              </div>
            )}
          </>
        )}

        <Button
          variant="ghost"
          className="w-full text-primary text-sm hover:bg-primary/5"
          onClick={handleViewHistory}
        >
          View All History
        </Button>
      </div>
    </div>
  );
}
