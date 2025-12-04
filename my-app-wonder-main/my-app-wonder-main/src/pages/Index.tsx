import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { Upload, Link as LinkIcon, Loader2 } from "lucide-react";

const Index = () => {
  const [imageUrl, setImageUrl] = useState<string>("");
  const [imagePreview, setImagePreview] = useState<string>("");
  const [classification, setClassification] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        toast({
          title: "Invalid file type",
          description: "Please upload an image file",
          variant: "destructive",
        });
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setImagePreview(result);
        setImageUrl(result);
        setClassification("");
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUrlSubmit = () => {
    if (!imageUrl.trim()) {
      toast({
        title: "URL required",
        description: "Please enter an image URL",
        variant: "destructive",
      });
      return;
    }
    setImagePreview(imageUrl);
    setClassification("");
  };

  const classifyImage = async () => {
    if (!imageUrl) {
      toast({
        title: "No image selected",
        description: "Please upload or provide an image URL first",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setClassification("");

    try {
      const { data, error } = await supabase.functions.invoke('classify-traffic-sign', {
        body: { imageUrl: imageUrl }
      });

      if (error) {
        throw error;
      }

      if (data.error) {
        throw new Error(data.error);
      }

      setClassification(data.classification);
      toast({
        title: "Classification complete",
        description: "Traffic sign identified successfully",
      });
    } catch (error) {
      console.error('Classification error:', error);
      toast({
        title: "Classification failed",
        description: error instanceof Error ? error.message : "Failed to classify image",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-secondary/20 to-background p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Traffic Sign Classifier
          </h1>
          <p className="text-muted-foreground">
            Upload or provide an image URL to identify traffic signs with AI-powered accuracy
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left Side - Image Input */}
          <Card className="p-6 space-y-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Upload Image</h2>
              <Tabs defaultValue="upload" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="upload">
                    <Upload className="w-4 h-4 mr-2" />
                    Upload File
                  </TabsTrigger>
                  <TabsTrigger value="url">
                    <LinkIcon className="w-4 h-4 mr-2" />
                    Image URL
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="upload" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="file-upload">Choose an image file</Label>
                    <Input
                      id="file-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="cursor-pointer"
                    />
                  </div>
                </TabsContent>
                
                <TabsContent value="url" className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="image-url">Enter image URL</Label>
                    <div className="flex gap-2">
                      <Input
                        id="image-url"
                        type="url"
                        placeholder="https://example.com/image.jpg"
                        value={imageUrl}
                        onChange={(e) => setImageUrl(e.target.value)}
                      />
                      <Button onClick={handleUrlSubmit} variant="secondary">
                        Load
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>

            {imagePreview && (
              <div className="space-y-4">
                <div className="rounded-lg overflow-hidden border border-border bg-muted/50">
                  <img
                    src={imagePreview}
                    alt="Selected traffic sign"
                    className="w-full h-auto max-h-96 object-contain"
                  />
                </div>
                <Button
                  onClick={classifyImage}
                  disabled={isLoading}
                  className="w-full"
                  size="lg"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Classify Traffic Sign"
                  )}
                </Button>
              </div>
            )}
          </Card>

          {/* Right Side - Classification Results */}
          <Card className="p-6 space-y-4">
            <h2 className="text-2xl font-semibold">Detection Results</h2>
            
            {!classification && !isLoading && (
              <div className="flex items-center justify-center h-64 text-muted-foreground">
                <div className="text-center">
                  <p className="text-lg mb-2">No results yet</p>
                  <p className="text-sm">Upload an image and click "Classify Traffic Sign" to see results</p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
                  <p className="text-muted-foreground">Analyzing traffic sign...</p>
                </div>
              </div>
            )}

            {classification && (
              <div className="prose prose-sm max-w-none">
                <div className="bg-muted/50 rounded-lg p-6 border border-border">
                  <pre className="whitespace-pre-wrap text-sm text-foreground font-sans">
                    {classification}
                  </pre>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;
