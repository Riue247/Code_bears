import { InputGroup,InputGroupAddon,InputGroupInput, } from "@/components/ui/input-group"
import { SearchIcon } from "lucide-react"
import { Item, ItemActions, ItemContent, ItemDescription, ItemTitle } from "../ui/item"
import { Button } from "../ui/button"
import { useState } from "react";
export default function Search({
    onSearch,
  }: {
    onSearch: (address: string) => Promise<void> | void;
  }) {
    const result = {name:"Baltimore City College", distance:1.2}
    const [query, setQuery] = useState("");
    return (
        <div className="flex flex-col gap-2 p-4">
            <InputGroup>
               <InputGroupInput placeholder="Search an address or place"
                 onChange={(e) => setQuery(e.target.value)}
                 onKeyDown={(e) => {
                    if (e.key === "Enter" && query.trim()) {
                      onSearch(query.trim());
                    }
                 }}
                 value={query}
               />
               <InputGroupAddon>
                <SearchIcon/>
               </InputGroupAddon>
            </InputGroup>
           <div>
            <Item className="border-2 border-zinc-200 dark:border-zinc-800 rounded-md">
                <ItemContent>
                    <ItemTitle>{result.name}</ItemTitle>
                    <ItemDescription>{result.distance} miles away</ItemDescription>
                </ItemContent>
                <ItemActions>
                    <Button variant="outline" disabled={!query.trim()} onClick={() => onSearch(query.trim())}>Go</Button>
                </ItemActions>
                
            </Item>
           </div>
        </div>
    )
}
