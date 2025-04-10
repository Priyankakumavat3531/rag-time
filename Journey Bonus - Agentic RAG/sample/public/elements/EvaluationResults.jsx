import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

export default function EvaluationResults() {
  const relevance = props.evaluations.find(e => e.type === 'relevance');
  const groundedness = props.evaluations.find(e => e.type === 'groundedness');

  // Utility to convert a score (1 to 5) into a percentage (width for bar)
  const progressWidth = score => `${(score / 5) * 100}%`
  return (
    <Card className="my-4">
      <CardHeader>
        <CardTitle>Evaluation Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-6">
          <div className="mb-1 font-medium text-lg">Groundedness</div>
          <div className="relative h-4 w-full bg-gray-200 rounded-md">
            <div
              className="absolute h-4 rounded-md bg-primary"
              style={{ width: progressWidth(groundedness.score) }}
            />
          </div>
          <div className="mt-1 text-sm text-gray-900">
            {groundedness.score} out of 5
          </div>
          <div className="mt-1 text-sm text-gray-900">
            {groundedness.explanation}
          </div>
        </div>
        <div>
          <div className="mb-1 font-medium text-lg">Relevance</div>
          <div className="relative h-4 w-full bg-gray-200 rounded-md">
            <div
              className="absolute h-4 rounded-md bg-primary"
              style={{ width: progressWidth(relevance.score) }}
            />
          </div>
          <div className="mt-1 text-sm text-gray-900">
            {relevance.score} out of 5
          </div>
          <div className="mt-1 text-sm text-gray-900">
            {relevance.explanation}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}