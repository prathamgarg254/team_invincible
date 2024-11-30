import { Dashboard } from "./pages/dashboard"
import {BrowserRouter,Route,Routes} from "react-router-dom"
import './index.css';

function App(){
  return <div className="" >
    <BrowserRouter>
    <Routes>
      <Route path={"/"} element={<Dashboard/>}/>
    </Routes>
    </BrowserRouter>
    
  </div>
}

export default App