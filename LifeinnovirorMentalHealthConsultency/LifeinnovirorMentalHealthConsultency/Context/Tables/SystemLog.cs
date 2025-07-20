using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class SystemLog
    {
        [Key]
        public int LogId { get; set; }

        [Required]
        public string ActorType { get; set; }

        [Required]
        public int ActorId { get; set; }

        [Required]
        public string Action { get; set; }

        public string Details { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.Now;
    }
}