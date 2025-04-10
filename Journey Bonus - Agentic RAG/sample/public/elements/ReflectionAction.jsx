import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { StopCircle, Globe, Pencil } from "lucide-react"

export default function CreateElement() {
    titles = {
        "stop": {
          "title": "🛑",
          "description": "Stop"
        },
        "web_search": {
          "title": "🌐",
          "description": "Web Search"
        },
        "index_search": {
          "title": "📝",
          "description": "Index Search"
        }
    }
    return (
        <Card className="my-4">
        <CardHeader>
          <CardTitle>Next Action: {titles[props.follow_up_action]['title']} {titles[props.follow_up_action]['description']}</CardTitle>
        </CardHeader>
        <CardContent>
            <div className="flex items-center">
              <span><b>Reason</b>: {props.follow_up_action_reason}</span>
            </div>
            <div className="flex items-center">
              <span><b>Answer Reflection</b>: {props.answer_reflection}</span>
            </div>
            {props.next_query &&
              <div className="flex items-center mb-2">
                <span><b>Next Query</b>: {props.next_query}</span>
              </div>
            }
        </CardContent>
      </Card>
    )
}