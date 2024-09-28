import { useNavigate } from "react-router-dom"

export function Header(){
    
    return <div>
        <div className="flex justify-between px-4 bg-black text-white h-12 items-center ">
            <div className="cursor-pointer">
                 connecting dots </div>
            <div className="flex justify-end ">
                <div className="px-1 cursor-pointer ont-mono">About</div>
                <div className="px-1 cursor-pointer">get Involved</div>
                <div className="px-1 cursor-pointer">contect</div>
            </div>
        </div>
    </div>
}